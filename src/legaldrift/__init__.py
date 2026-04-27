"""LegalDrift: Statistical monitoring for legal document revision.

LegalDrift detects when the substantive meaning of a legal document shifts
between versions. It is designed for lawyers, contract managers, and compliance
officers who need to verify that a redlined contract, an updated policy, or a
renegotiated agreement has not introduced unintended semantic changes.

This library performs offline, two-sample drift detection using non-parametric
statistical tests (Kolmogorov-Smirnov, Mann-Whitney U, Maximum Mean Discrepancy,
Energy Distance) combined via Fisher's method. It does not learn from labeled
examples and does not require a training corpus.

For full documentation, see the README and the docs/ directory.
"""

__version__ = "0.1.0"
__author__ = "OsamaMoftah"

from legaldrift.core.document import LegalDocument
from legaldrift.core.embedding import EmbeddingEngine
from legaldrift.core.detector import DriftDetector
from legaldrift.core.concepts import LegalConceptExtractor
from legaldrift.core.chunker import (
    DocumentChunk,
    chunk_by_paragraphs,
    chunk_by_sections,
    chunk_by_sentences,
    align_chunks,
)
from legaldrift.core.history import DriftHistory, DriftRecord

__all__ = [
    "LegalDocument",
    "EmbeddingEngine",
    "DriftDetector",
    "LegalConceptExtractor",
    "DocumentChunk",
    "chunk_by_paragraphs",
    "chunk_by_sections",
    "chunk_by_sentences",
    "align_chunks",
    "DriftHistory",
    "DriftRecord",
]
