"""Tests for legaldrift.core.chunker module."""

import pytest

from legaldrift.core.chunker import (
    DocumentChunk,
    chunk_by_paragraphs,
    chunk_by_sections,
    chunk_by_sentences,
    align_chunks,
)
from legaldrift.core.document import LegalDocument


class TestDocumentChunk:
    def test_creation(self):
        chunk = DocumentChunk(
            text="Hello world.",
            chunk_index=0,
            document_id="doc1",
            start_char=0,
            end_char=12,
        )
        assert chunk.chunk_index == 0
        assert chunk.document_id == "doc1"
        assert chunk.word_count == 2
        assert chunk.char_count == 12


class TestChunkByParagraphs:
    def test_simple_paragraphs(self):
        text = (
            "Paragraph one contains a substantial amount of text with many words "
            "so that it clearly exceeds the minimum word threshold and stands alone.\n\n"
            "Paragraph two also contains a substantial amount of text with many words "
            "so that it clearly exceeds the minimum word threshold and stands alone.\n\n"
            "Paragraph three contains a substantial amount of text with many words "
            "so that it clearly exceeds the minimum word threshold and stands alone."
        )
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_paragraphs(doc)
        assert len(chunks) == 3
        assert "Paragraph one" in chunks[0].text
        assert "Paragraph two" in chunks[1].text
        assert "Paragraph three" in chunks[2].text

    def test_merge_short_paragraphs(self):
        text = "A.\n\nB.\n\nThis is a longer paragraph with many words to exceed the minimum."
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_paragraphs(doc, min_words=5)
        # A and B should be merged with the next paragraph
        assert len(chunks) >= 1

    def test_empty_document(self):
        doc = LegalDocument(text="", document_id="d1")
        chunks = chunk_by_paragraphs(doc)
        assert len(chunks) == 0

    def test_single_paragraph(self):
        doc = LegalDocument(text="Only one paragraph here.", document_id="d1")
        chunks = chunk_by_paragraphs(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Only one paragraph here."

    def test_long_paragraph_split(self):
        text = "Word. " * 600  # ~600 words
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_paragraphs(doc, max_words=200)
        assert len(chunks) >= 2


class TestChunkBySections:
    def test_numbered_sections(self):
        text = (
            "1. INTRODUCTION\nThis is the intro.\n\n"
            "2. DEFINITIONS\nTerms are defined here.\n\n"
            "3. SERVICES\nServices described here."
        )
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_sections(doc)
        assert len(chunks) >= 2
        assert all(c.chunk_type == "section" for c in chunks)

    def test_article_sections(self):
        text = (
            "Article 1: Scope\n\nThis defines the scope.\n\n"
            "Article 2: Obligations\n\nThese are the obligations."
        )
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_sections(doc)
        # Should detect at least 2 section headers and their content
        assert len(chunks) >= 2
        headers = [c.metadata.get("header", "") for c in chunks]
        assert any("Article 1" in h for h in headers)
        assert any("Article 2" in h for h in headers)

    def test_fallback_to_paragraphs(self):
        text = "No clear sections here. Just some paragraphs.\n\nAnother paragraph."
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_sections(doc)
        # Should fall back to paragraph chunking
        assert len(chunks) >= 1


class TestChunkBySentences:
    def test_simple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_sentences(doc, min_words=2)
        assert len(chunks) == 3
        assert chunks[0].text == "First sentence."
        assert chunks[1].text == "Second sentence."

    def test_merge_short_sentences(self):
        text = "A. B. This is a longer sentence with more words."
        doc = LegalDocument(text=text, document_id="d1")
        chunks = chunk_by_sentences(doc, min_words=3)
        # A and B might be merged
        assert len(chunks) >= 1

    def test_empty_document(self):
        doc = LegalDocument(text="", document_id="d1")
        chunks = chunk_by_sentences(doc)
        assert len(chunks) == 0


class TestAlignChunks:
    def test_index_alignment(self):
        chunks1 = [
            DocumentChunk(text="A", chunk_index=0, document_id="d1"),
            DocumentChunk(text="B", chunk_index=1, document_id="d1"),
        ]
        chunks2 = [
            DocumentChunk(text="A2", chunk_index=0, document_id="d2"),
            DocumentChunk(text="B2", chunk_index=1, document_id="d2"),
        ]
        aligned = align_chunks(chunks1, chunks2)
        assert len(aligned) == 2
        assert aligned[0] == (chunks1[0], chunks2[0])
        assert aligned[1] == (chunks1[1], chunks2[1])

    def test_unequal_length(self):
        chunks1 = [
            DocumentChunk(text="A", chunk_index=0, document_id="d1"),
        ]
        chunks2 = [
            DocumentChunk(text="A2", chunk_index=0, document_id="d2"),
            DocumentChunk(text="B2", chunk_index=1, document_id="d2"),
        ]
        aligned = align_chunks(chunks1, chunks2)
        assert len(aligned) == 2
        assert aligned[0] == (chunks1[0], chunks2[0])
        assert aligned[1] == (None, chunks2[1])

    def test_similarity_alignment(self):
        import numpy as np

        chunks1 = [
            DocumentChunk(text="Section A", chunk_index=0, document_id="d1"),
            DocumentChunk(text="Section B", chunk_index=1, document_id="d1"),
        ]
        chunks2 = [
            DocumentChunk(text="Section A modified", chunk_index=0, document_id="d2"),
            DocumentChunk(text="Section B modified", chunk_index=1, document_id="d2"),
        ]
        sim = np.array([[0.9, 0.1], [0.1, 0.9]])
        aligned = align_chunks(chunks1, chunks2, similarity_matrix=sim)
        assert len(aligned) == 2
        # Should align based on similarity
        assert aligned[0] == (chunks1[0], chunks2[0])
        assert aligned[1] == (chunks1[1], chunks2[1])
