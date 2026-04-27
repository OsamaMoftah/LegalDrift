"""Tests for legaldrift top-level imports and CLI."""

import pytest


def test_top_level_imports():
    """Verify all public API classes are importable."""
    from legaldrift import LegalDocument, EmbeddingEngine, DriftDetector, LegalConceptExtractor

    assert LegalDocument is not None
    assert EmbeddingEngine is not None
    assert DriftDetector is not None
    assert LegalConceptExtractor is not None


def test_core_imports():
    from legaldrift.core import (
        LegalDocument,
        SourceReliability,
        EmbeddingEngine,
        DriftDetector,
        DriftResult,
        LegalConceptExtractor,
        ADWIN,
        DDM,
        HDP,
        BaselineResult,
    )

    assert DriftResult is not None
    assert SourceReliability is not None
    assert BaselineResult is not None
