# LegalDrift

[![CI](https://github.com/OsamaMoftah/LegalDrift/actions/workflows/ci.yml/badge.svg)](https://github.com/OsamaMoftah/LegalDrift/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/legaldrift.svg?v=0.1.0)](https://pypi.org/project/legaldrift/)
[![Python Versions](https://img.shields.io/pypi/pyversions/legaldrift.svg?v=0.1.0)](https://pypi.org/project/legaldrift/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Statistical monitoring for legal document revision.**

LegalDrift detects when the substantive meaning of a legal document shifts between versions. It is designed for lawyers, contract managers, and compliance officers who need to verify that a redlined contract, an updated policy, or a renegotiated agreement has not introduced unintended semantic changes.

This is not a replacement for legal review. It is a screening tool to flag passages that merit closer human attention.

---

## Table of Contents

- [When to Use LegalDrift](#when-to-use-legaldrift)
- [When Not to Use It](#when-not-to-use-it)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Command-Line Usage](#command-line-usage)
- [Python API](#python-api)
- [Localized Drift Detection](#localized-drift-detection)
- [Drift History & Audit Logs](#drift-history--audit-logs)
- [Concept Extraction](#concept-extraction)
- [Statistical Methods](#statistical-methods)
- [Limitations & Caveats](#limitations--caveats)
- [Contributing](#contributing)
- [License](#license)

---

## When to Use LegalDrift

You might find this useful if you:

- Review multiple redline rounds of the same contract and want a sanity check that nothing was silently broadened or narrowed.
- Maintain a template library and need to verify that a "minor update" has not altered the substantive scope of a standard clause.
- Track regulatory compliance documents (e.g., privacy policies, AI governance frameworks) across jurisdictions where a wording tweak can change legal effect.
- Compare an executed agreement against the final draft to confirm no last-minute substitutions occurred.

## When Not to Use It

LegalDrift is intentionally narrow in scope. It will **not** help you with:

- **Determining whether a change is legally valid or enforceable.** It flags that text has shifted; it does not opine on whether the shift is permissible.
- **Detecting purely formatting or numbering changes.** It operates on semantic embeddings, not on character-level diffs. A renumbered Section 3.2 that says the exact same thing will not trigger a drift alert.
- **Short, low-entropy text.** A one-sentence NDA addendum will not yield meaningful statistical results.
- **Cross-lingual comparison.** The current embedding model is English-centric.
- **Finding typos or grammatical errors.** These are invisible to the semantic layer.

If you need a traditional redline tool, use a document comparison suite (e.g., Workshare, Litera Compare, or your document editor's built-in comparison). LegalDrift complements those tools; it does not replace them.

---

## Installation

```bash
pip install legaldrift
```

For development:

```bash
git clone https://github.com/OsamaMoftah/LegalDrift.git
cd legaldrift
pip install -e ".[dev]"
pytest
```

### Requirements

- Python 3.8 or later
- `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`
- `sentence-transformers` (optional; required for the Legal-BERT embedding backend)

If `sentence-transformers` is unavailable, LegalDrift falls back to a deterministic hash-based embedding. The fallback is faster and offline, but less sensitive to nuanced semantic shifts. For production use, we recommend installing the optional dependency.

---

## Quick Start

### 1. Detect drift between two contracts

```bash
legaldrift detect contract_v1.txt contract_v2.txt
```

Example output (using the bundled sample contracts):

```
Drift Detection Results
========================================
Drift Detected: True
P-value: 0.0012
Confidence: 99.88%
Severity: 0.2417
Effect Size: 0.8421

Individual Tests:
  ks_test: p=0.0034
  mannwhitney: p=0.0011
  mmd: p=0.0008
  energy: p=0.0029
```

A low p-value (typically < 0.05) indicates that the two documents occupy measurably different regions of the embedding space. In plain English: the second version says something meaningfully different from the first.

### 2. Analyze a single document for legal concepts

```bash
legaldrift analyze contract.txt
```

Output:

```
Document Analysis
========================================
Document ID: contract
Jurisdiction: US
Word Count: 1,247
Character Count: 8,932

Legal Concepts Detected:
  - data_protection
  - obligation
  - permission
  - transparency
```

### 3. Compare section by section

```bash
legaldrift chunks contract_v1.txt contract_v2.txt
```

Output:

```
Chunked Drift Detection Results
============================================================
🔴 [DRIFT] 3. PAYMENT
    p=0.0034, severity=0.2417

🟢 [OK] 1. DEFINITIONS
    p=0.8912, severity=0.0124

🟡 [NEW] 10. AI GOVERNANCE
    added in current document
```

This tells you *where* to look, not just *whether* something changed.

---

## How It Works

LegalDrift converts each document into a high-dimensional vector using a language model trained on legal text (Legal-BERT). It then treats the collection of vectors from each document as a statistical distribution and asks: are these two distributions the same, or have they diverged?

Four non-parametric tests are run in parallel:

1. **Kolmogorov-Smirnov** — compares the overall shape of the distributions.
2. **Mann-Whitney U** — tests for location shifts (mean or median drift).
3. **Maximum Mean Discrepancy (MMD)** — a kernel-based measure of distributional distance.
4. **Energy Distance** — a geometric measure of separation between sample clouds.

The four p-values are combined via Fisher's method. The result is a single p-value and a severity score. There is no machine-learning classifier; there are no training labels. This is pure statistical hypothesis testing, which means the results are interpretable and the false-positive rate is controlled by your chosen threshold (default: 0.05).

For a deeper technical explanation, see [docs/architecture.md](docs/architecture.md).

---

## Command-Line Usage

LegalDrift exposes a single CLI entry point with subcommands.

### Global flags

| Flag | Description |
|------|-------------|
| `-j, --jurisdiction` | Default jurisdiction tag (US, EU, DE, UK, etc.) |
| `--no-legal-bert` | Force the hash-based fallback embedder |
| `-v, --verbose` | Debug logging |

### Subcommands

| Subcommand | Purpose |
|------------|---------|
| `detect` | Full-document drift test between two files |
| `analyze` | Concept extraction and metadata for one file |
| `chunks` | Section-by-section drift test |
| `compare` | Run LegalDrift alongside baseline methods (ADWIN, DDM, HDP) |
| `history` | Query saved drift records |

### Examples

```bash
# Detect with EU jurisdiction tag
legaldrift -j EU detect privacy_2024.txt privacy_2025.txt

# Save the result to an audit log
legaldrift detect v1.txt v2.txt \
  --history drift.db \
  --notes "Post-AI Act update" \
  --tags gdpr ai-act eu

# Query prior detections
legaldrift history --path drift.db --drift-only --limit 20
```

---

## Python API

### Full-document comparison

```python
from legaldrift import LegalDocument, EmbeddingEngine, DriftDetector

doc1 = LegalDocument(
    text=open("contract_v1.txt").read(),
    document_id="2024-001",
    jurisdiction="DE"
)

doc2 = LegalDocument(
    text=open("contract_v2.txt").read(),
    document_id="2024-001-r1",
    jurisdiction="DE"
)

engine = EmbeddingEngine()
detector = DriftDetector(threshold=0.05)

emb1 = engine.encode([doc1.text])
emb2 = engine.encode([doc2.text])

result = detector.detect(emb1, emb2)

print(f"Drift: {'YES' if result.drift_detected else 'NO'}")
print(f"p-value: {result.p_value:.4f}")
print(f"Severity: {result.severity:.4f}")
```

### Section-level comparison

```python
from legaldrift import chunk_by_sections, align_chunks

chunks1 = chunk_by_sections(doc1)
chunks2 = chunk_by_sections(doc2)

for c1, c2 in align_chunks(chunks1, chunks2):
    if c1 is None:
        print(f"[ADDED] {c2.metadata.get('header', 'Section')}")
        continue
    if c2 is None:
        print(f"[REMOVED] {c1.metadata.get('header', 'Section')}")
        continue

    e1 = engine.encode([c1.text])
    e2 = engine.encode([c2.text])
    r = detector.detect(e1, e2)

    if r.drift_detected:
        print(f"[DRIFT] Section {c1.chunk_index}: p={r.p_value:.4f}")
    else:
        print(f"[OK]    Section {c1.chunk_index}")
```

### Drift history and audit trails

```python
from legaldrift import DriftHistory

history = DriftHistory(path="audit.db", backend="sqlite")

history.save(
    baseline_id="2024-001",
    current_id="2024-001-r1",
    result=result,
    notes="Reviewed by J. Smith, 15 Jan 2025",
    tags=["ai-act", "high-risk"]
)

# Retrieve all drift-positive records
records = history.query(drift_detected=True, limit=50)
for r in records:
    print(r.timestamp, r.baseline_id, r.result["p_value"])
```

---

## Localized Drift Detection

Full-document comparison has a well-known weakness: a large, mostly unchanged contract can dilute a significant change in one clause. LegalDrift addresses this by splitting documents into semantically coherent chunks (paragraphs, sections, or sentences) and comparing each pair.

Three chunking strategies are available:

- `chunk_by_paragraphs()` — respects paragraph boundaries; merges very short paragraphs.
- `chunk_by_sections()` — splits on numbered or titled headers (e.g., "1. DEFINITIONS", "Article 3"); falls back to paragraph chunking if no headers are found.
- `chunk_by_sentences()` — finest granularity; useful for short agreements or isolated clauses.

Chunk alignment is index-based by default. If you have a custom similarity matrix (e.g., from a clause-matching preprocessor), you can pass it to `align_chunks()` for greedy nearest-neighbor alignment.

---

## Drift History & Audit Logs

Legal and compliance workflows require reproducibility. LegalDrift can persist every detection run to a JSON file or an SQLite database, along with:

- Timestamp (UTC)
- Baseline and current document IDs
- Full test results (p-values, severity, effect size)
- Human-readable notes
- Tags for categorization

This turns the tool from an ad-hoc script into a lightweight audit system. The SQLite backend supports indexed queries by document ID, drift status, date range, and tags.

---

## Concept Extraction

LegalDrift includes a lightweight regex-based extractor that flags common legal concept classes without requiring a heavy NLP pipeline:

| Concept | Example trigger phrases |
|---------|------------------------|
| `obligation` | "shall", "must", "is required" |
| `permission` | "may", "is permitted", "has the right" |
| `prohibition` | "shall not", "must not", "is prohibited" |
| `data_protection` | "GDPR", "personal data", "privacy" |
| `high_risk` | "high risk", "conformity assessment" |
| `automated_decision` | "automated decision", "algorithmic processing" |
| `transparency` | "transparency", "explainable AI" |
| `human_oversight` | "human oversight", "meaningful human control" |

The extractor is deliberately simple. It is intended as a first-pass triage tool, not as a substitute for a trained legal analyst or a full clause-ontology system.

---

## Statistical Methods

LegalDrift combines four complementary tests. Each test is sensitive to a different kind of distributional shift:

| Test | What it detects | Notes |
|------|----------------|-------|
| **Kolmogorov-Smirnov** | Shape differences in cumulative distributions | Non-parametric; works across all PCA-reduced dimensions |
| **Mann-Whitney U** | Location shifts (median/mean drift) | Robust to outliers; also run across all dimensions |
| **MMD** | General distributional divergence in kernel space | Computationally heavier; permutation-based p-value |
| **Energy Distance** | Geometric separation of point clouds | Related to the Wasserstein metric; intuitive geometric interpretation |

**Fisher's method** combines the four p-values into a single chi-squared statistic. This meta-test gains power when multiple individual tests show weak but consistent signals, and it remains valid even if one test is misspecified.

The p-value is not a probability that "the contract changed." It is the probability of observing this much separation between the two embedding distributions *if* the underlying semantic content had remained identical. A low p-value is therefore evidence against the null hypothesis of "no semantic drift."

---

## Limitations & Caveats

We believe tools that touch legal documents owe their users an honest account of what they cannot do.

1. **Statistical, not legal, significance.** A p-value of 0.01 means the embeddings are different. It does not mean the difference is legally material. A shifted comma in a damages cap can be legally catastrophic yet statistically invisible; a stylistic rewrite of a boilerplate clause can be legally trivial yet statistically detectable.

2. **Embedding bias.** Legal-BERT was trained on a corpus of US and EU legal text. Its semantic space may not accurately represent non-Western legal traditions, highly technical scientific agreements, or domain-specific jargon (e.g., maritime salvage law, biotech licensing).

3. **Single-sample documents.** When comparing one document to one document, the detector has limited statistical power. Chunking improves this, but the tool is most reliable when you have multiple samples per version (e.g., a corpus of standard-form contracts from 2024 versus 2025).

4. **No temporal modeling.** The baselines named ADWIN, DDM, and HDP are included for comparative benchmarking. They are not implemented as true streaming change detectors; they are offline two-sample approximations.

5. **No format parsing.** LegalDrift expects plain text. You must extract text from PDF, Word, or HTML before ingestion.

If your use case involves high-stakes transactional work or regulatory submissions, treat LegalDrift as a **screening layer**, not a sign-off layer.

---

## Contributing

We welcome contributions from legal informaticists, data scientists, and practitioners. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, testing requirements, and our code of conduct.

If you are a legal professional with no programming background but a concrete use case, please open a GitHub Issue. User stories are as valuable as pull requests.

---

## License

MIT License. See [LICENSE](LICENSE) for the full text.

LegalDrift is provided as-is, without warranty of any kind. The authors and contributors assume no liability for decisions made on the basis of its output. Always consult a qualified legal professional before acting on contract analysis.
