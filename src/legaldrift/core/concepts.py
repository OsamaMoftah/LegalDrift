"""Legal concept extraction from documents.

Pattern-based extraction of legal concepts like obligations, permissions, etc.
"""

import logging
import re
from typing import Dict, List, Set

from legaldrift.core.document import LegalDocument

logger = logging.getLogger(__name__)


class LegalConceptExtractor:
    """Extracts legal concepts from document text using regex patterns."""

    PATTERNS = {
        'automated_decision': [
            r"automated.{0,20}decision",
            r"algorithmic.{0,20}decision",
            r"automatic.{0,20}processing",
            r"ai.{0,20}decision",
        ],
        'data_protection': [
            r"data protection",
            r"privacy",
            r"gdpr",
            r"personal.{0,10}data",
        ],
        'transparency': [
            r"transparency",
            r"explainability",
            r"explainable.{0,10}ai",
        ],
        'high_risk': [
            r"high.{0,10}risk",
            r"critical.{0,10}system",
            r"conformity.{0,10}assessment",
        ],
        'human_oversight': [
            r"human.{0,10}oversight",
            r"human.{0,10}supervision",
            r"meaningful.{0,10}human.{0,10}control",
        ],
        'obligation': [
            r"shall",
            r"must",
            r"is required",
            r" obligated",
        ],
        'permission': [
            r"may",
            r"is permitted",
            r"is allowed",
            r"has the right",
        ],
        'prohibition': [
            r"shall not",
            r"must not",
            r"is prohibited",
            r"may not",
        ],
    }

    def __init__(self):
        """Initialize the extractor with compiled patterns."""
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for concept, pattern_list in self.PATTERNS.items():
            self.compiled_patterns[concept] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]

    def extract(self, documents: List[LegalDocument]) -> Set[str]:
        """Extract concepts from a list of documents.

        Args:
            documents: List of LegalDocument objects.

        Returns:
            Set of concept names found in the documents.
        """
        found = set()

        for doc in documents:
            text = doc.text.lower()
            for concept, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        found.add(concept)
                        break

        logger.info("Extracted %d concepts from %d documents", len(found), len(documents))
        return found

    def extract_from_text(self, text: str) -> Set[str]:
        """Extract concepts from raw text.

        Args:
            text: Document text.

        Returns:
            Set of concept names found.
        """
        text_lower = text.lower()
        found = set()

        for concept, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    found.add(concept)
                    break

        return found

    def get_concept_counts(self, documents: List[LegalDocument]) -> Dict[str, int]:
        """Count occurrences of each concept across documents."""
        counts: Dict[str, int] = {}

        for doc in documents:
            concepts = self.extract_from_text(doc.text)
            for concept in concepts:
                counts[concept] = counts.get(concept, 0) + 1

        return counts