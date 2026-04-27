"""Core module containing the main components."""

from legaldrift.core.document import LegalDocument, SourceReliability
from legaldrift.core.embedding import EmbeddingEngine
from legaldrift.core.detector import DriftDetector, DriftResult
from legaldrift.core.concepts import LegalConceptExtractor
from legaldrift.core.baselines import ADWIN, DDM, HDP, BaselineResult
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
    "SourceReliability",
    "EmbeddingEngine",
    "DriftDetector",
    "DriftResult",
    "LegalConceptExtractor",
    "ADWIN",
    "DDM",
    "HDP",
    "BaselineResult",
    "DocumentChunk",
    "chunk_by_paragraphs",
    "chunk_by_sections",
    "chunk_by_sentences",
    "align_chunks",
    "DriftHistory",
    "DriftRecord",
]
