"""Legal document data model.

Core data structures for representing legal documents and their metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SourceReliability:
    """Reliability rating for a document source."""

    score: float = 0.5  # 0.0 to 1.0
    verified: bool = False
    source_type: str = "unknown"  # e.g., "court", "legislature", "contract"

    def __post_init__(self):
        self.score = max(0.0, min(1.0, float(self.score)))


@dataclass
class LegalDocument:
    """Represents a legal document with metadata.

    Attributes:
        text: The full text content of the document.
        document_id: Unique identifier for the document.
        jurisdiction: Legal jurisdiction (e.g., "US", "EU", "DE").
        metadata: Additional key-value metadata.
        source_reliability: Reliability information about the source.
    """

    text: str
    document_id: str = "unknown"
    jurisdiction: str = "US"
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_reliability: Optional[SourceReliability] = None

    def __post_init__(self):
        if self.source_reliability is None:
            self.source_reliability = SourceReliability()

    @property
    def word_count(self) -> int:
        """Return the number of words in the document."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Return the number of characters in the document."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document to a dictionary."""
        return {
            "document_id": self.document_id,
            "jurisdiction": self.jurisdiction,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
            "source_reliability": {
                "score": self.source_reliability.score if self.source_reliability else 0.5,
                "verified": self.source_reliability.verified if self.source_reliability else False,
                "source_type": (
                    self.source_reliability.source_type if self.source_reliability else "unknown"
                ),
            },
            "text_preview": self.text[:500] + "..." if len(self.text) > 500 else self.text,
        }
