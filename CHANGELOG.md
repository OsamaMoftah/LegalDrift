# Changelog

All notable changes to this project are documented here.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the format recommended by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **Localized drift detection.** Documents can now be split into paragraphs, sections, or sentences, and each chunk compared independently. This addresses the dilution problem where a single changed clause in a long contract was previously invisible to the full-document detector.
- **Drift history and audit logging.** Results can be persisted to JSON or SQLite, with indexed querying by document ID, drift status, date range, and tags. This supports compliance workflows that require reproducible records.
- **Multivariate statistical tests.** The Kolmogorov-Smirnov and Mann-Whitney U tests now run across all PCA-reduced dimensions and aggregate p-values via Fisher's method, rather than testing only the first principal component.
- **Jurisdiction tagging.** The CLI and Python API now accept a jurisdiction flag (US, EU, DE, UK, etc.) for document metadata. Previously hardcoded to "US".
- **Sample contracts.** Two realistic SaaS agreement variants (`contract_v1.txt` and `contract_v2.txt`) are included in `src/data/sample/` for immediate testing.
- **GitHub Actions CI.** Automated testing across Ubuntu, macOS, and Windows on Python 3.8 through 3.12.
- **Jupyter tutorial.** `notebooks/tutorial.ipynb` walks through drift detection, concept extraction, chunking, and history usage.
- **Issue and PR templates.** Standardized bug reports, feature requests, and pull request checklists.

### Fixed

- **Missing core module.** The `LegalDocument` and `SourceReliability` classes were referenced by every module but the `document.py` file itself was absent, rendering the package completely non-importable.
- **Package naming inconsistency.** Three different names (`legal-drift`, `legal_drift`, `legaldrift`) were used across `pyproject.toml`, imports, and directory names. These have been unified to `legaldrift`.
- **Single-sample PCA crash.** Comparing one document against one document triggered a `ValueError` because PCA requested more components than available samples. The fix caps components at `total_samples - 1`.
- **NaN effect size.** Cohen's d returned `nan` when both samples had zero variance (common with single-sample embeddings). It now returns `0.0` in that edge case.
- **pyproject.toml entry points.** The CLI console script pointed to a non-existent module path.

### Changed

- **SHA-256 hashing.** The hash-based fallback embedder previously used MD5, which is cryptographically broken and looked unprofessional in a security-sensitive legal tool. Replaced with SHA-256.
- **CLI expansion.** New subcommands: `chunks` (section-level comparison) and `history` (query saved records). The `detect` command now supports `--history`, `--notes`, and `--tags` flags.

## [0.1.0] — 2024-01-15

### Added

- Initial public release.
- Offline two-sample drift detection via KS, Mann-Whitney U, MMD, and Energy Distance, combined by Fisher's method.
- Legal-BERT embedding backend with a deterministic hash fallback.
- Regex-based legal concept extraction for obligations, permissions, prohibitions, GDPR references, AI Act references, and more.
- Baseline comparison methods: ADWIN, DDM, and HDP (implemented as offline approximations for benchmarking).
- CLI with `detect`, `analyze`, and `compare` subcommands.
- Streamlit demo application (`demo/app.py`).
