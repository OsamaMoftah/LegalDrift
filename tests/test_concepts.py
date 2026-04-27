"""Tests for legaldrift.core.concepts module."""

import pytest

from legaldrift.core.concepts import LegalConceptExtractor
from legaldrift.core.document import LegalDocument


class TestLegalConceptExtractor:
    def test_extract_obligation(self):
        extractor = LegalConceptExtractor()
        text = "The party shall deliver the goods by Monday."
        concepts = extractor.extract_from_text(text)
        assert "obligation" in concepts

    def test_extract_permission(self):
        extractor = LegalConceptExtractor()
        text = "The user may access the system at any time."
        concepts = extractor.extract_from_text(text)
        assert "permission" in concepts

    def test_extract_prohibition(self):
        extractor = LegalConceptExtractor()
        text = "The licensee shall not redistribute the software."
        concepts = extractor.extract_from_text(text)
        assert "prohibition" in concepts

    def test_extract_gdpr(self):
        extractor = LegalConceptExtractor()
        text = "This policy complies with GDPR requirements for personal data."
        concepts = extractor.extract_from_text(text)
        assert "data_protection" in concepts

    def test_extract_high_risk(self):
        extractor = LegalConceptExtractor()
        text = "This is a high risk system requiring conformity assessment."
        concepts = extractor.extract_from_text(text)
        assert "high_risk" in concepts

    def test_extract_multiple(self):
        extractor = LegalConceptExtractor()
        text = (
            "The controller shall ensure transparency. "
            "The user may request access. "
            "Automated decision making is prohibited."
        )
        concepts = extractor.extract_from_text(text)
        assert "obligation" in concepts
        assert "permission" in concepts
        assert "transparency" in concepts
        assert "automated_decision" in concepts

    def test_no_concepts(self):
        extractor = LegalConceptExtractor()
        text = "The quick brown fox jumps over the lazy dog."
        concepts = extractor.extract_from_text(text)
        assert len(concepts) == 0

    def test_extract_from_documents(self):
        extractor = LegalConceptExtractor()
        docs = [
            LegalDocument(text="You shall comply with all regulations.", document_id="d1"),
            LegalDocument(text="We may terminate this agreement.", document_id="d2"),
        ]
        concepts = extractor.extract(docs)
        assert "obligation" in concepts
        assert "permission" in concepts

    def test_get_concept_counts(self):
        extractor = LegalConceptExtractor()
        docs = [
            LegalDocument(text="You shall comply.", document_id="d1"),
            LegalDocument(text="You shall also report monthly.", document_id="d2"),
            LegalDocument(text="We may terminate.", document_id="d3"),
        ]
        counts = extractor.get_concept_counts(docs)
        assert counts.get("obligation", 0) == 2
        assert counts.get("permission", 0) == 1

    def test_case_insensitive(self):
        extractor = LegalConceptExtractor()
        text = "THE PARTY SHALL DELIVER."
        concepts = extractor.extract_from_text(text)
        assert "obligation" in concepts
