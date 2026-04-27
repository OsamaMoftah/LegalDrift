# Architecture & Design Decisions

This document explains how LegalDrift works under the hood, and why specific design choices were made. It is written for technically inclined users and contributors who want to understand the engineering trade-offs.

## Overview

LegalDrift is an **offline, two-sample drift detector** for legal text. It does not learn from labeled examples, and it does not maintain a running window of data. Instead, it treats each document (or chunk) as a random sample from an underlying semantic distribution, and asks whether two such distributions are statistically distinguishable.

## The Pipeline

```
Raw Text
    |
    v
[Chunker]  ──>  DocumentChunk[]
    |
    v
[Embedding Engine]  ──>  np.ndarray (n_chunks × d_embedding)
    |
    v
[Statistical Tests]  ──>  p-values[]
    |
    v
[Fisher's Method]  ──>  Combined p-value, severity, effect size
    |
    v
[History / CLI / API]
```

## 1. Chunking

### Why chunk at all?

A full-document embedding averages over the entire text. A one-paragraph change in a 50-page merger agreement can be drowned out by 49 pages of identical boilerplate. Chunking lets us localize the signal.

### Design choices

- **Paragraph chunking** is the default. It respects the author's own semantic boundaries and is robust across document styles.
- **Section chunking** attempts to detect headers ("1. DEFINITIONS", "Article 3", etc.). If no headers are found, it falls back to paragraph chunking rather than failing.
- **Sentence chunking** is available for very short documents, but we do not recommend it for contracts because individual sentences often lack enough embedding variance to yield stable p-values.

### Merge thresholds

Very short paragraphs (signature blocks, single-line cross-references) are merged with the next paragraph. Very long paragraphs are split at sentence boundaries. These thresholds are configurable but were chosen empirically: chunks below ~20 words produce unstable embeddings, and chunks above ~500 words lose localization.

## 2. Embeddings

### Legal-BERT

The default embedding model is `nlpaueb/legal-bert-base-uncased`, a BERT model fine-tuned on US case law and EU legislative text. It was selected because:

- It is small enough to run on a CPU ( ~100 MB).
- It outperforms general-domain BERT on legal semantic similarity benchmarks.
- It is freely available and permissively licensed.

### Hash fallback

If `sentence-transformers` is unavailable (e.g., offline CI, restricted environments), LegalDrift falls back to a deterministic hash-based embedder. This embedder:

- Splits text into words.
- Hashes each word into a fixed-size vector via SHA-256.
- Upweights words that match a small, hand-curated list of legal terms ("shall", "gdpr", "liability", etc.).
- Adds small Gaussian noise to break ties.
- L2-normalizes the result.

The hash fallback is **not** semantically aware in the way a transformer is. "Tenant" and "Lessee" will land in completely different hash buckets. However, it is reproducible, fast, offline, and sufficient for detecting coarse vocabulary shifts (e.g., the sudden appearance of "AI Act" terminology in a 2023 privacy policy).

### Why not OpenAI / Cohere / etc.?

Third-party API embeddings were considered and rejected for the core library because:

1. **Privacy.** Sending client contracts to an external API may violate confidentiality obligations or data-processing agreements.
2. **Cost.** Per-token pricing makes batch analysis of large document collections expensive.
3. **Reproducibility.** API models are updated silently; a detection run in January may not be reproducible in February.
4. **Offline operation.** Many legal professionals work in air-gapped or VPN-restricted environments.

API-based embeddings may be added as an optional plugin in the future.

## 3. Dimensionality Reduction

The raw embedding dimension is 768 (Legal-BERT base). Running four multivariate tests on 768-dimensional vectors with small sample sizes is numerically unstable and slow. We therefore apply PCA to reduce the dimensionality before the KS and Mann-Whitney tests.

- **Target:** 10 components.
- **Hard ceiling:** Never request more components than `min(n_samples, n_features, total_samples - 1)`. This prevents the single-sample crash that was present in early versions.
- **Why PCA?** It preserves global variance structure, which is what the distributional tests care about. It is deterministic (no random seed issues) and fast.

MMD and Energy Distance are run on the full embeddings because they are less sensitive to high-dimensional variance collapse.

## 4. Statistical Tests

### Why four tests?

No single test is uniformly most powerful against all alternatives. KS is sensitive to shape changes; Mann-Whitney to location shifts; MMD to general distributional divergence; Energy Distance to geometric separation. Running all four and combining them via Fisher's method gives us:

- **Power.** A shift that is weak in one test metric may be strong in another.
- **Robustness.** If one test is misspecified (e.g., due to skewed chunk lengths), the others still contribute valid evidence.
- **Interpretability.** The individual p-values are reported, so a user can see *which* test drove the signal.

### Fisher's method

Fisher's method transforms k p-values into a chi-squared statistic:

```
χ² = -2 Σ ln(p_i)
df = 2k
```

Under the null hypothesis (no drift), this statistic follows a chi-squared distribution with 2k degrees of freedom. The combined p-value is the tail probability of this distribution.

**Caveat:** Fisher's method assumes the individual tests are independent. In practice, KS and Mann-Whitney on the same PCA-reduced data are positively correlated. This makes the combined p-value slightly anti-conservative (i.e., it may drift-detect more often than the nominal threshold). We accept this trade-off because LegalDrift is a screening tool, not a confirmatory test. Users who need strict family-wise error control should apply a Bonferroni correction manually.

### Permutation testing for MMD and Energy Distance

MMD and Energy Distance do not have closed-form null distributions for arbitrary embeddings. We therefore estimate p-values via permutation:

1. Compute the observed statistic on the original (baseline, current) pairing.
2. Pool the two samples.
3. Randomly reshuffle the pooled sample into two new groups of the original sizes.
4. Recompute the statistic.
5. Repeat 200 times (configurable).
6. The p-value is the proportion of permutation statistics that equal or exceed the observed statistic.

This is computationally expensive but assumption-free.

## 5. Severity and Effect Size

- **Severity:** Wasserstein distance between the first PCA components of the two samples. It measures how far apart the distributions are in units of the data, not in probability space.
- **Effect size:** Cohen's d, a standardized mean difference. A rule of thumb: 0.2 is small, 0.5 is medium, 0.8 is large. However, in high-dimensional embedding space, Cohen's d can be misleading because the "mean" is not a human-interpretable quantity.

Both metrics are reported for transparency, but we recommend relying on the combined p-value and the individual test breakdown for decision-making.

## 6. History & Persistence

### JSON backend

A simple append-only JSON file. Good for personal workflows, single-user scripts, and environments where SQLite is unavailable. Not suitable for concurrent writes.

### SQLite backend

A relational schema with indexed columns on `timestamp`, `baseline_id`, and `drift_detected`. Supports concurrent reads and basic querying. The schema is intentionally flat: we store the full result blob as JSON in a single column rather than normalizing every test statistic into its own table. This keeps the schema stable even as we add or remove individual tests.

### Why not a full document management system?

Because LegalDrift is a library, not an application. We store *metadata and results*, not the documents themselves. The user is free to integrate LegalDrift into their existing DMS, ELN, or contract lifecycle management platform via the Python API.

## 7. Baselines (ADWIN, DDM, HDP)

These three methods are included for benchmarking and educational purposes. They are **not** true streaming implementations:

- **ADWIN** (Adaptive Windowing) normally operates on a sequence of error rates over time. Our version treats the baseline and current embeddings as a single pooled window and applies the ADWIN cut heuristic.
- **DDM** (Drift Detection Method) normally tracks classification error rates. Our version tracks distance-from-centroid as a proxy for error.
- **HDP** (Hierarchical Dirichlet Process) normally models topic evolution over time. Our version uses K-means on PCA-reduced embeddings as a crude topic proxy.

These simplifications are documented in the source code. They are useful for sanity-checking LegalDrift's results against established concept-drift literature, but they should not be used as primary detectors without understanding the approximations involved.

## 8. Concept Extraction

The `LegalConceptExtractor` is intentionally primitive. It uses compiled regular expressions to flag concept classes. This design choice was deliberate:

- **No training data required.** We do not need a labeled corpus of legal clauses.
- **Deterministic.** The same text always yields the same concepts.
- **Transparent.** A lawyer can read the regex list and understand exactly what is being flagged.

The downside is that it cannot handle paraphrase, negation scope, or syntactic variation (e.g., "The obligation to indemnify shall not apply if..." will match both `obligation` and `prohibition`). For advanced extraction, users should integrate a dedicated NLP pipeline (e.g., spaCy with a legal NER model) and pass its output to LegalDrift as pre-computed tags.

---

## References

- Chouldechova, A., & Roth, A. (2020). *A Snapshot of the Frontiers of Fairness in Machine Learning*. Communications of the ACM.
- Fisher, R. A. (1948). *Combining Independent Tests of Significance*. American Statistical Association.
- Ipek, P. (2020). *The use of BERT in legal text classification*. Artificial Intelligence and Law.
- Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika. (Original DDM paper)
- Bifet, A., & Gavalda, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing*. SIAM International Conference on Data Mining. (Original ADWIN paper)
