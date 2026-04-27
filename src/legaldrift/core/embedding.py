"""Embedding engine for legal document vectorization.

Supports Legal-BERT and fallback hash-based embeddings.
"""

import hashlib
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Converts text documents into vector embeddings.

    Uses Legal-BERT when available, falls back to hash-based embeddings.
    """

    LEGAL_TERMS = {
        'gdpr', 'privacy', 'consent', 'data', 'protection', 'automated',
        'decision', 'transparency', 'obligation', 'ai', 'high_risk',
        'compensation', 'termination', 'confidentiality', 'non-compete',
        'liability', 'indemnification', 'warranty', 'jurisdiction'
    }

    def __init__(
        self,
        use_legal_bert: bool = True,
        embedding_dim: int = 384,
        rng: Optional[np.random.Generator] = None
    ):
        """Initialize the embedding engine.

        Args:
            use_legal_bert: Whether to use Legal-BERT (requires internet).
            embedding_dim: Dimension of embeddings for fallback mode.
            rng: Random number generator for reproducibility.
        """
        self.use_legal_bert = use_legal_bert
        self.embedding_dim = embedding_dim
        self.rng = rng or np.random.default_rng()
        self.model = None

        if self.use_legal_bert:
            self._init_legal_bert()

    def _init_legal_bert(self):
        """Initialize Legal-BERT model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading Legal-BERT: nlpaueb/legal-bert-base-uncased")
            self.model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")
            logger.info("Legal-BERT loaded successfully")
        except Exception as e:
            logger.warning("Legal-BERT unavailable: %s. Using hash embeddings.", e)
            self.use_legal_bert = False

    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for processing.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        if self.use_legal_bert and self.model is not None:
            return self._encode_legal_bert(texts, batch_size)
        return self._encode_hash(texts)

    def _encode_legal_bert(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using Legal-BERT."""
        cleaned = [t.strip() if isinstance(t, str) and t.strip() else "empty" for t in texts]

        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def _encode_hash(self, texts: List[str]) -> np.ndarray:
        """Fallback hash-based encoding."""
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        for i, text in enumerate(texts):
            words = set(text.lower().split())

            for word in words:
                h = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.embedding_dim
                weight = 5.0 if word in self.LEGAL_TERMS else 1.0
                embeddings[i, h] += weight

            embeddings[i] += self.rng.normal(0, 0.1, self.embedding_dim)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        return embeddings

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))