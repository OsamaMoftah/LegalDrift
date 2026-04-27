"""Tests for legaldrift.core.detector module."""

import numpy as np
import pytest

from legaldrift.core.detector import DriftDetector, DriftResult


class TestDriftResult:
    def test_to_dict(self):
        result = DriftResult(
            drift_detected=True,
            p_value=0.01,
            confidence=0.99,
            severity=0.5,
            effect_size=0.3,
            threshold=0.05,
            tests={"ks_test": {"statistic": 0.1, "p_value": 0.02}},
        )
        d = result.to_dict()
        assert d["drift_detected"] is True
        assert d["p_value"] == 0.01
        assert d["tests"]["ks_test"]["p_value"] == 0.02


class TestDriftDetector:
    def test_no_drift_identical_distributions(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 10))
        current = rng.normal(0, 1, (100, 10))
        detector = DriftDetector(threshold=0.05, rng=rng)
        result = detector.detect(baseline, current)
        assert isinstance(result, DriftResult)
        # Identical distributions should not show strong drift
        assert result.p_value > 0.01

    def test_drift_different_distributions(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 10))
        current = rng.normal(5, 1, (100, 10))  # Strong mean shift
        detector = DriftDetector(threshold=0.05, rng=rng)
        result = detector.detect(baseline, current)
        # Strong shift should be detected
        assert result.p_value < 0.1
        assert result.drift_detected is True
        assert result.severity > 0

    def test_result_fields_populated(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (50, 5))
        current = rng.normal(0, 1, (50, 5))
        detector = DriftDetector(threshold=0.05, rng=rng)
        result = detector.detect(baseline, current)
        assert isinstance(result.p_value, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.severity, float)
        assert isinstance(result.effect_size, float)
        assert result.threshold == 0.05
        assert "ks_test" in result.tests
        assert "mannwhitney" in result.tests
        assert "mmd" in result.tests
        assert "energy" in result.tests

    def test_small_samples(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (10, 3))
        current = rng.normal(0, 1, (10, 3))
        detector = DriftDetector(rng=rng)
        result = detector.detect(baseline, current)
        assert isinstance(result, DriftResult)

    def test_fisher_combine_all_high_p(self):
        detector = DriftDetector()
        p_values = [0.9, 0.8, 0.7, 0.85]
        chi2, p_combined = detector._fisher_combine(p_values)
        assert p_combined > 0.5

    def test_fisher_combine_all_low_p(self):
        detector = DriftDetector()
        p_values = [0.01, 0.02, 0.015, 0.03]
        chi2, p_combined = detector._fisher_combine(p_values)
        assert p_combined < 0.05

    def test_fisher_combine_zeros(self):
        detector = DriftDetector()
        p_values = [0.0, 0.0]
        chi2, p_combined = detector._fisher_combine(p_values)
        assert p_combined == 0.0

    def test_confidence_range(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (50, 5))
        current = rng.normal(0, 1, (50, 5))
        detector = DriftDetector(rng=rng)
        result = detector.detect(baseline, current)
        assert 0.0 < result.confidence < 1.0

    def test_cohens_d_symmetry(self):
        detector = DriftDetector()
        x = np.array([[1], [2], [3]])
        y = np.array([[4], [5], [6]])
        d1 = detector._cohens_d(x, y)
        d2 = detector._cohens_d(y, x)
        assert pytest.approx(abs(d1), 0.001) == abs(d2)
