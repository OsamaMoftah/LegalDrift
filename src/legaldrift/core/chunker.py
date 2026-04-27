"""Document chunking for localized drift detection.

Splits long legal documents into semantic chunks (paragraphs, sections)
to enable detection of *which* part of a document changed.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from legaldrift.core.document import LegalDocument


@dataclass
class DocumentChunk:
    """A chunk of a legal document with metadata."""

    text: str
    chunk_index: int
    document_id: str
    start_char: int = 0
    end_char: int = 0
    chunk_type: str = "paragraph"  # "paragraph", "section", "sentence"
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Return word count of this chunk."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Return character count of this chunk."""
        return len(self.text)


def chunk_by_paragraphs(
    doc: LegalDocument, min_words: int = 20, max_words: int = 500
) -> List[DocumentChunk]:
    """Split a document into paragraph-level chunks.

    Paragraphs shorter than min_words are merged with the next paragraph.
    Paragraphs longer than max_words are split into sub-paragraphs.

    Args:
        doc: LegalDocument to chunk.
        min_words: Minimum words per chunk.
        max_words: Maximum words per chunk.

    Returns:
        List of DocumentChunk objects.
    """
    # Split on double newlines or numbered section headers
    raw_paragraphs = re.split(r"\n\s*\n|\n(?=\s*\d+\.\s+)", doc.text.strip())
    chunks: List[DocumentChunk] = []
    current_text = ""
    current_start = 0
    idx = 0

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        words = len(para.split())

        if words < min_words and not current_text:
            # Start accumulating small paragraphs
            current_text = para
            current_start = doc.text.find(para, current_start)
            continue

        if current_text:
            para = current_text + "\n\n" + para
            words = len(para.split())
            current_text = ""

        if words > max_words:
            # Split long paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_text = ""
            sub_start = current_start if current_start else doc.text.find(para)
            for sent in sentences:
                if len((sub_text + " " + sent).split()) > max_words and sub_text:
                    end = sub_start + len(sub_text)
                    chunks.append(
                        DocumentChunk(
                            text=sub_text.strip(),
                            chunk_index=idx,
                            document_id=doc.document_id,
                            start_char=sub_start,
                            end_char=end,
                            chunk_type="paragraph",
                        )
                    )
                    idx += 1
                    sub_start = end + 1
                    sub_text = sent
                else:
                    sub_text = (sub_text + " " + sent).strip() if sub_text else sent
            if sub_text:
                end = sub_start + len(sub_text)
                chunks.append(
                    DocumentChunk(
                        text=sub_text.strip(),
                        chunk_index=idx,
                        document_id=doc.document_id,
                        start_char=sub_start,
                        end_char=end,
                        chunk_type="paragraph",
                    )
                )
                idx += 1
        else:
            start = doc.text.find(para, current_start)
            end = start + len(para)
            chunks.append(
                DocumentChunk(
                    text=para,
                    chunk_index=idx,
                    document_id=doc.document_id,
                    start_char=start,
                    end_char=end,
                    chunk_type="paragraph",
                )
            )
            idx += 1
            current_start = end

    if current_text:
        start = doc.text.find(current_text, current_start)
        end = start + len(current_text)
        chunks.append(
            DocumentChunk(
                text=current_text,
                chunk_index=idx,
                document_id=doc.document_id,
                start_char=start,
                end_char=end,
                chunk_type="paragraph",
            )
        )

    return chunks


def chunk_by_sections(doc: LegalDocument) -> List[DocumentChunk]:
    """Split a document by numbered or titled sections.

    Recognizes patterns like:
      1. Section Title
      Article 3
      SECTION 1: DEFINITIONS

    Args:
        doc: LegalDocument to chunk.

    Returns:
        List of DocumentChunk objects, one per section.
    """
    # Match common section headers
    section_pattern = re.compile(
        r"(?:^|\n\n)\s*(?:"
        r"(?:SECTION|Article|ARTICLE)\s*\d+[.:]?\s*[^\n]*"
        r"|\d+\.\s+[A-Z][^\n]{3,200}"
        r"|[A-Z][A-Z\s]{3,60}[.:]\s*$"
        r")",
        re.MULTILINE | re.IGNORECASE,
    )

    matches = list(section_pattern.finditer(doc.text))
    if len(matches) < 2:
        # Fall back to paragraph chunking if no clear sections
        return chunk_by_paragraphs(doc)

    chunks: List[DocumentChunk] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)
        text = doc.text[start:end].strip()
        header = match.group(0).strip()[:100]
        chunks.append(
            DocumentChunk(
                text=text,
                chunk_index=i,
                document_id=doc.document_id,
                start_char=start,
                end_char=end,
                chunk_type="section",
                metadata={"header": header},
            )
        )

    return chunks


def chunk_by_sentences(doc: LegalDocument, min_words: int = 10) -> List[DocumentChunk]:
    """Split a document into sentence-level chunks.

    Useful for fine-grained drift detection.

    Args:
        doc: LegalDocument to chunk.
        min_words: Minimum words per chunk (shorter sentences merged).

    Returns:
        List of DocumentChunk objects.
    """
    sentences = re.split(r"(?<=[.!?])\s+", doc.text)
    chunks: List[DocumentChunk] = []
    current = ""
    current_start = 0
    idx = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current:
            current = current + " " + sent
        else:
            current = sent
            current_start = doc.text.find(sent, current_start)

        if len(current.split()) >= min_words:
            end = current_start + len(current)
            chunks.append(
                DocumentChunk(
                    text=current,
                    chunk_index=idx,
                    document_id=doc.document_id,
                    start_char=current_start,
                    end_char=end,
                    chunk_type="sentence",
                )
            )
            idx += 1
            current_start = end + 1
            current = ""

    if current:
        end = current_start + len(current)
        chunks.append(
            DocumentChunk(
                text=current,
                chunk_index=idx,
                document_id=doc.document_id,
                start_char=current_start,
                end_char=end,
                chunk_type="sentence",
            )
        )

    return chunks


def align_chunks(
    chunks1: List[DocumentChunk],
    chunks2: List[DocumentChunk],
    similarity_matrix: Optional[np.ndarray] = None,
) -> List[Tuple[Optional[DocumentChunk], Optional[DocumentChunk]]]:
    """Align two chunk lists by index, with optional similarity-based re-ordering.

    Default is index-based alignment (chunk 0 with chunk 0, etc.).
    For documents with inserted/deleted sections, a custom similarity matrix
    can be used to find best matches.

    Args:
        chunks1: Chunks from document 1.
        chunks2: Chunks from document 2.
        similarity_matrix: Optional (n1 x n2) similarity matrix for custom alignment.

    Returns:
        List of aligned chunk pairs. None indicates a missing chunk.
    """
    if similarity_matrix is not None:
        # Greedy alignment based on similarity
        n1, n2 = similarity_matrix.shape
        aligned: List[Tuple[Optional[DocumentChunk], Optional[DocumentChunk]]] = []
        used2 = set()

        for i in range(n1):
            best_j = None
            best_sim = -1.0
            for j in range(n2):
                if j in used2:
                    continue
                if similarity_matrix[i, j] > best_sim:
                    best_sim = similarity_matrix[i, j]
                    best_j = j
            if best_j is not None and best_sim > 0.3:
                aligned.append((chunks1[i], chunks2[best_j]))
                used2.add(best_j)
            else:
                aligned.append((chunks1[i], None))

        for j in range(n2):
            if j not in used2:
                aligned.append((None, chunks2[j]))

        return aligned

    # Simple index-based alignment
    max_len = max(len(chunks1), len(chunks2))
    aligned = []
    for i in range(max_len):
        c1 = chunks1[i] if i < len(chunks1) else None
        c2 = chunks2[i] if i < len(chunks2) else None
        aligned.append((c1, c2))
    return aligned
