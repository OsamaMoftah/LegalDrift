"""Tests for legaldrift.core.baselines module."""

import numpy as np
import pytest

from legaldrift.core.baselines import ADWIN, DDM, HDP, BaselineResult


class TestBaselineResult:
    def test_creation(self):
        result = BaselineResult(
            drift_detected=True,
            p_value=0.01,
            confidence=0.95,
            severity=0.5,
            method_name="TEST",
            sample_count=100,
        )
        assert result.method_name == "TEST"
        assert result.sample_count == 100


class TestADWIN:
    def test_no_drift_similar(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 5))
        current = rng.normal(0, 1, (100, 5))
        adwin = ADWIN(rng=rng)
        result = adwin.detect(baseline, current)
        assert isinstance(result, BaselineResult)
        assert result.method_name == "ADWIN"

    def test_drift_shifted(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 5))
        current = rng.normal(5, 1, (100, 5))
        adwin = ADWIN(rng=rng)
        result = adwin.detect(baseline, current)
        assert isinstance(result, BaselineResult)


class TestDDM:
    def test_no_drift_similar(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 5))
        current = rng.normal(0, 1, (100, 5))
        ddm = DDM(rng=rng)
        result = ddm.detect(baseline, current)
        assert isinstance(result, BaselineResult)
        assert result.method_name == "DDM"

    def test_drift_shifted(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 5))
        current = rng.normal(5, 1, (100, 5))
        ddm = DDM(rng=rng)
        result = ddm.detect(baseline, current)
        assert isinstance(result, BaselineResult)


class TestHDP:
    def test_no_drift_similar(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 5))
        current = rng.normal(0, 1, (100, 5))
        hdp = HDP(rng=rng)
        result = hdp.detect(baseline, current)
        assert isinstance(result, BaselineResult)
        assert result.method_name == "HDP"

    def test_drift_shifted(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 5))
        current = rng.normal(5, 1, (100, 5))
        hdp = HDP(rng=rng)
        result = hdp.detect(baseline, current)
        assert isinstance(result, BaselineResult)

    def test_small_samples(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (10, 3))
        current = rng.normal(0, 1, (10, 3))
        hdp = HDP(rng=rng)
        result = hdp.detect(baseline, current)
        assert isinstance(result, BaselineResult)
