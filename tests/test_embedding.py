"""Tests for legaldrift.core.embedding module."""

import numpy as np
import pytest

from legaldrift.core.embedding import EmbeddingEngine


class TestEmbeddingEngine:
    def test_hash_fallback_initialization(self):
        engine = EmbeddingEngine(use_legal_bert=False, embedding_dim=128)
        assert engine.use_legal_bert is False
        assert engine.embedding_dim == 128
        assert engine.model is None

    def test_hash_fallback_encoding(self):
        engine = EmbeddingEngine(use_legal_bert=False, embedding_dim=64)
        texts = ["This is a privacy clause.", "Obligations and termination."]
        embeddings = engine.encode(texts)
        assert embeddings.shape == (2, 64)
        assert embeddings.dtype == np.float32

    def test_hash_empty_list(self):
        engine = EmbeddingEngine(use_legal_bert=False)
        result = engine.encode([])
        assert result.size == 0

    def test_hash_legal_terms_weighted(self):
        engine = EmbeddingEngine(use_legal_bert=False, embedding_dim=100)
        # Text with legal terms should produce different embeddings than generic text
        emb_legal = engine.encode(["gdpr privacy data protection"])
        emb_generic = engine.encode(["hello world foo bar"])
        assert emb_legal.shape == (1, 100)
        assert emb_generic.shape == (1, 100)
        # They should not be identical
        assert not np.allclose(emb_legal, emb_generic, atol=0.1)

    def test_hash_reproducibility(self):
        engine1 = EmbeddingEngine(
            use_legal_bert=False, embedding_dim=64, rng=np.random.default_rng(42)
        )
        engine2 = EmbeddingEngine(
            use_legal_bert=False, embedding_dim=64, rng=np.random.default_rng(42)
        )
        text = ["confidentiality and non-compete liability."]
        emb1 = engine1.encode(text)
        emb2 = engine2.encode(text)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_normalization(self):
        engine = EmbeddingEngine(use_legal_bert=False, embedding_dim=32)
        embeddings = engine.encode(["some text here"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.array([1.0]))

    def test_compute_similarity_identical(self):
        engine = EmbeddingEngine(use_legal_bert=False, embedding_dim=32)
        emb = engine.encode(["test text"])
        sim = engine.compute_similarity(emb[0], emb[0])
        assert pytest.approx(sim, 0.001) == 1.0

    def test_compute_similarity_different(self):
        engine = EmbeddingEngine(use_legal_bert=False, embedding_dim=256)
        emb = engine.encode(["gdpr privacy", "hello world"])
        sim = engine.compute_similarity(emb[0], emb[1])
        assert -1.0 <= sim <= 1.0  # Cosine similarity range
        assert sim < 1.0  # Different texts should not be identical

    def test_compute_similarity_zero(self):
        engine = EmbeddingEngine(use_legal_bert=False, embedding_dim=32)
        sim = engine.compute_similarity(np.zeros(32), np.ones(32))
        assert sim == 0.0
