# Contributing to LegalDrift

LegalDrift sits at the intersection of law and statistical computing. That means our contributors often come from one domain with limited exposure to the other. That is fine. In fact, it is the point. What matters is intellectual honesty and respect for the constraints of both fields.

This document covers how to set up the project, how to write code that will be accepted, and how to report issues in a way that leads to useful fixes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing Requirements](#testing-requirements)
- [Code Style](#code-style)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)
- [Security & Data Handling](#security--data-handling)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Community Standards](#community-standards)

---

## Prerequisites

You will need:

- Python 3.8 or later
- `git`
- A working familiarity with pytest (if you are writing code)
- A working familiarity with legal documents (if you are filing issues or testing edge cases)

No contribution is too small. Typo fixes, clearer docstrings, and additional test cases are all genuinely useful.

---

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork:

   ```bash
   git clone https://github.com/OsamaMoftah/LegalDrift.git
   cd legaldrift
   ```

3. Install in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

4. Create a branch:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

---

## Development Workflow

### Running Tests

Run the full test suite before any commit:

```bash
pytest
```

For coverage:

```bash
pytest --cov=legaldrift
```

All tests must pass. We do not merge PRs with failing CI, and we do not accept "I will fix the tests later" commitments.

If you are adding a new statistical test or a new chunking strategy, include tests that demonstrate:

- Correct behavior on synthetic data with known drift.
- Correct behavior on synthetic data with *no* drift (the null case is as important as the alternative case).
- Graceful handling of edge cases: empty documents, single-word documents, mismatched chunk counts, and non-ASCII characters.

### Adding Tests

Tests live in `tests/` and are named `test_<module>.py`. Use descriptive names:

```python
def test_mann_whitney_returns_high_p_for_identical_distributions():
    ...
```

Avoid abbreviations in test names. A test that fails in CI six months from now should be readable by someone who has never seen the codebase.

---

## Code Style

We use [Black](https://github.com/psf/black) with a line length of 100 characters.

```bash
black src/ tests/
```

Docstrings should explain *why*, not merely *what*. A docstring that reads "Detects drift" is worse than no docstring. A good docstring reads:

> "Detects distributional drift between two embeddings. This is an offline, two-sample test; it does not model temporal drift and should not be used for streaming data without modification."

Type hints are required on all public functions. Internal helper functions may omit them if the logic is trivial, but err on the side of clarity.

---

## Reporting Bugs

A good bug report saves hours of maintainer time. Please include:

1. **What you did.** A minimal code snippet or CLI invocation.
2. **What you expected.** The output or behavior you thought would occur.
3. **What actually happened.** The full traceback, if any, and the actual output.
4. **Your environment.** Python version, operating system, and whether `sentence-transformers` was installed.
5. **The document text (or a proxy).** If the bug involves a specific contract or clause, include a minimal, anonymized excerpt that reproduces the issue. Do not paste confidential client material into a public GitHub issue.

If you are legally prohibited from sharing the document, create a synthetic paragraph that triggers the same behavior.

---

## Proposing Features

Feature requests are welcome, but we are skeptical of scope creep. LegalDrift is intentionally narrow. Before opening a feature request, ask yourself:

- Does this belong in the core library, or would it be better as a downstream script or plugin?
- Does it preserve the interpretability of the output, or does it introduce a black-box model?
- Is the use case specific enough that a lawyer or contract manager could explain it to a colleague in one sentence?

If the answer to all three is yes, open an issue with:

- A concrete user story ("As a compliance officer, I need to ...")
- The current workaround, if any
- A sketch of the proposed API or CLI interface

We will discuss it in the open. There is no guarantee of acceptance, but there is a guarantee of a reasoned response.

---

## Security & Data Handling

LegalDrift processes text that may be privileged, confidential, or subject to regulatory protection. Contributors must respect the following rules:

- **No logging of document text.** The logger may emit metadata (word counts, file paths, drift p-values), but it must never write document text to stdout, stderr, or log files.
- **No network transmission in core modules.** The embedding engine downloads Legal-BERT once at initialization, but the detector, chunker, and history modules are strictly offline. Any feature that introduces a cloud API call belongs in a separate optional package.
- **No persistent storage of raw text.** The `DriftHistory` class stores detection results, timestamps, and tags. It does not store the original document text unless explicitly configured to do so by the user.
- **Secure defaults.** Hash-based embeddings use SHA-256, not MD5. Random number generators should accept an optional `seed` or `rng` parameter so that results are reproducible.

If you discover a security vulnerability, please email the maintainers directly rather than opening a public issue. We will acknowledge receipt within 48 hours and provide a timeline for a fix.

---

## Commit Messages

Write commit messages in the imperative mood, as if commanding the codebase to change:

```
feat: add paragraph-level chunking with min/max word thresholds

Documents with very short paragraphs (e.g., signature blocks,
enumerated lists) were previously creating noise in the drift
signal. This commit merges short paragraphs and splits long ones
to keep chunks in a statistically useful size range.
```

Allowed prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or correcting tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `perf:` — performance improvement
- `chore:` — build process, dependency updates, CI

---

## Pull Request Process

1. Ensure all tests pass and code is formatted with Black.
2. Update `CHANGELOG.md` under the `[Unreleased]` heading. Follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.
3. If your change affects the public API or CLI, update the relevant sections of `README.md` or `docs/`.
4. Reference any related issue(s) in the PR description.
5. Keep the diff focused. A PR that changes one thing is easier to review than a PR that changes five things.
6. Be prepared for direct, constructive criticism. We review code, not people.

Once your PR is approved and CI passes, a maintainer will merge it. We do not squash-merge without the author's consent.

---

## Community Standards

We expect participants to:

- Assume good faith, even when disagreement is sharp.
- Cite sources when making claims about legal concepts, statistical theory, or empirical results.
- Respect that legal practice varies across jurisdictions. A feature that seems obvious in US contract law may be irrelevant or misleading in civil-law jurisdictions.

Harassment, ad hominem attacks, or dismissive behavior toward non-technical legal contributors will result in immediate removal from the project spaces.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
