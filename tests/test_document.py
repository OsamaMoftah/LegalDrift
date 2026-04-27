"""Tests for legaldrift.core.document module."""

import pytest

from legaldrift.core.document import LegalDocument, SourceReliability


class TestSourceReliability:
    def test_default_values(self):
        sr = SourceReliability()
        assert sr.score == 0.5
        assert sr.verified is False
        assert sr.source_type == "unknown"

    def test_score_clamping(self):
        sr = SourceReliability(score=1.5)
        assert sr.score == 1.0

        sr = SourceReliability(score=-0.3)
        assert sr.score == 0.0

    def test_custom_values(self):
        sr = SourceReliability(score=0.9, verified=True, source_type="court")
        assert sr.score == 0.9
        assert sr.verified is True
        assert sr.source_type == "court"


class TestLegalDocument:
    def test_basic_creation(self):
        doc = LegalDocument(text="This is a contract.", document_id="doc_1")
        assert doc.document_id == "doc_1"
        assert doc.jurisdiction == "US"
        assert doc.word_count == 4
        assert doc.char_count == 19

    def test_word_count_empty(self):
        doc = LegalDocument(text="")
        assert doc.word_count == 0
        assert doc.char_count == 0

    def test_jurisdiction_custom(self):
        doc = LegalDocument(text="Text.", jurisdiction="EU")
        assert doc.jurisdiction == "EU"

    def test_metadata(self):
        doc = LegalDocument(text="Text.", metadata={"author": "Alice", "date": "2024-01-01"})
        assert doc.metadata["author"] == "Alice"

    def test_source_reliability_default(self):
        doc = LegalDocument(text="Text.")
        assert doc.source_reliability is not None
        assert doc.source_reliability.score == 0.5

    def test_to_dict(self):
        doc = LegalDocument(text="Hello world.", document_id="test_doc", jurisdiction="DE")
        d = doc.to_dict()
        assert d["document_id"] == "test_doc"
        assert d["jurisdiction"] == "DE"
        assert d["word_count"] == 2
        assert d["char_count"] == 12
        assert "source_reliability" in d
        assert "text_preview" in d

    def test_to_dict_long_text(self):
        long_text = "x" * 1000
        doc = LegalDocument(text=long_text)
        d = doc.to_dict()
        assert d["text_preview"].endswith("...")
        assert len(d["text_preview"]) == 503  # 500 + "..."
