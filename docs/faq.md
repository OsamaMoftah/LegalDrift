# Frequently Asked Questions

## General

### Is LegalDrift a replacement for a lawyer?

No. It is a statistical screening tool. It can flag that two versions of a document are semantically divergent, but it cannot tell you whether that divergence is legally permissible, commercially sensible, or compliant with regulation. Always consult a qualified legal professional before acting on contract analysis.

### Do I need machine learning expertise to use this?

No. The CLI is designed to be usable by anyone who can run a terminal command. The Python API requires basic Python literacy. You do not need to understand embeddings or p-values to use the tool, though reading the [architecture documentation](architecture.md) will help you interpret the results responsibly.

### Is my document text sent to the cloud?

Not by default. The core library is entirely offline. It downloads the Legal-BERT model once (approximately 100 MB) and then processes everything locally. If you choose to use a cloud-based embedding model in the future, that will be an optional plugin, not the default behavior.

## Usage

### What file formats are supported?

Plain text (`.txt`) is supported natively. PDF, Word, and HTML are not parsed by LegalDrift. You must extract text first using a tool like `pdftotext`, `pandoc`, or your document editor's "Save as Text" feature.

### Can I compare more than two documents?

Not in a single CLI invocation. The `detect` and `chunks` commands take exactly two files. However, you can script multiple pairwise comparisons using the Python API, or batch-process a directory with a short shell loop.

### Can I compare a contract against a template?

Yes, and this is one of the most useful workflows. Save your master template as `template.txt` and each executed variant as `variant_001.txt`, `variant_002.txt`, etc. Then run:

```bash
for f in variant_*.txt; do
  legaldrift detect template.txt "$f" --history audit.json
done
```

### What does a p-value of 0.03 mean in practice?

It means that, if the two documents were semantically identical and the divergence was purely due to random noise in the embedding process, you would expect to see this much separation only 3% of the time. In other words, the observed separation is statistically unusual under the "no drift" hypothesis.

It does **not** mean there is a 97% probability that the contract changed. P-values are not probabilities of hypotheses.

### Why did I get "Drift Detected: False" when I can see obvious changes?

Possible reasons:

1. **The changes are stylistic, not semantic.** Replacing "shall" with "must" everywhere may not shift the embedding enough to register.
2. **The document is too short.** With very few chunks, the statistical tests lack power.
3. **The threshold is too conservative.** The default threshold is 0.05. You can relax it with `--threshold 0.10`, but be aware that this increases the false-positive rate.
4. **You are using the hash fallback.** The hash embedder is much less sensitive to nuanced semantic shifts than Legal-BERT. Install `sentence-transformers` for better results.

### Why did I get "Drift Detected: True" when the documents look identical?

Possible reasons:

1. **Formatting artifacts.** Hidden characters, encoding differences, or PDF extraction noise can create spurious word tokens.
2. **Header/footer drift.** Page numbers, dates, or file names in headers can shift the vocabulary distribution.
3. **False positives.** At a 0.05 threshold, you expect 1 false positive in every 20 comparisons of truly identical documents. If you run hundreds of comparisons, some will drift-detect by chance alone.
4. **PCA instability.** With very small documents (one or two chunks), the PCA step can produce unstable components. Try chunking the document into smaller pieces or adding more text.

## Technical

### Can I use a different embedding model?

Yes, at the API level. The `EmbeddingEngine` class accepts an optional model path or HuggingFace identifier. However, the CLI currently hardcodes Legal-BERT. If you need a custom model in the CLI, open an issue or submit a PR.

### Can I add my own legal concepts to the extractor?

Yes. The `LegalConceptExtractor` class stores its patterns in a dictionary that you can override or extend at runtime:

```python
from legaldrift.core.concepts import LegalConceptExtractor

extractor = LegalConceptExtractor()
extractor.PATTERNS["force_majeure"] = [r"force majeure", r"act of god"]
extractor.compiled_patterns = {
    k: [re.compile(p, re.IGNORECASE) for p in v]
    for k, v in extractor.PATTERNS.items()
}
```

A more robust plugin system is on the roadmap.

### How do I interpret the individual test p-values?

Look for consistency. If all four tests show low p-values, the drift signal is strong and multifaceted. If only one test is significant, the drift may be specific to that test's sensitivity (e.g., Mann-Whitney detecting a location shift but not a shape change). If the tests contradict each other sharply, treat the result with skepticism and inspect the documents manually.

### Is the combined p-value corrected for multiple testing?

No. Fisher's method is a meta-test, not a multiple-testing correction like Bonferroni or Benjamini-Hochberg. It gains power by pooling evidence across tests, but it does not control the family-wise error rate in the strict sense. If you need strict FWER control (e.g., for a published research paper), apply Bonferroni manually by dividing your threshold by 4.

### Can I run this in a CI pipeline?

Yes. The test suite runs in under 15 seconds. The CLI is deterministic when you use the hash fallback and set a fixed random seed. For CI, we recommend:

```bash
pip install legaldrift
legaldrift --no-legal-bert detect baseline.txt candidate.txt
```

The `--no-legal-bert` flag avoids downloading the transformer model in ephemeral CI environments.
