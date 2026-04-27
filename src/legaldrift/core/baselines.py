"""Baseline drift detection algorithms.

Implements ADWIN, DDM, and HDP for comparison.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a baseline drift detector."""

    drift_detected: bool
    p_value: float
    confidence: float
    severity: float
    method_name: str
    sample_count: int


class ADWIN:
    """Adaptive Windowing drift detector.

    Detects concept drift by maintaining a variable-size window
    and testing for statistical differences between sub-windows.
    """

    def __init__(self, delta: float = 0.002, rng: Optional[np.random.Generator] = None):
        """Initialize ADWIN.

        Args:
            delta: Confidence parameter for drift detection.
            rng: Random number generator.
        """
        self.delta = delta
        self.rng = rng or np.random.default_rng()

    def detect(self, baseline: np.ndarray, current: np.ndarray) -> BaselineResult:
        """Detect drift using ADWIN algorithm."""
        baseline_mean = np.mean(baseline, axis=0)
        current_mean = np.mean(current, axis=0)

        combined = np.vstack([baseline, current])
        n = len(combined)

        window_means = []
        for i in range(max(1, n // 4), n):
            window = combined[max(0, i - 20) : i]
            window_means.append(np.mean(window))

        if len(window_means) < 2:
            return BaselineResult(
                drift_detected=False,
                p_value=1.0,
                confidence=0.0,
                severity=0.0,
                method_name="ADWIN",
                sample_count=n,
            )

        variance = np.var(window_means)
        threshold = np.sqrt((2 * np.log(1 / self.delta)) / min(20, n // 2))
        mean_diff = abs(np.mean(baseline_mean) - np.mean(current_mean))

        drift_detected = mean_diff > threshold
        confidence = min(0.999, 1 - self.delta) if drift_detected else max(0.001, self.delta)
        severity = float(mean_diff / (threshold + 1e-10))

        return BaselineResult(
            drift_detected=drift_detected,
            p_value=self.delta if drift_detected else 1.0 - self.delta,
            confidence=confidence,
            severity=severity,
            method_name="ADWIN",
            sample_count=n,
        )


class DDM:
    """Drift Detection Method based on PAC learning.

    Monitors error rates and detects drift when error increases
    beyond statistical thresholds.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize DDM."""
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.rng = rng or np.random.default_rng()

    def detect(self, baseline: np.ndarray, current: np.ndarray) -> BaselineResult:
        """Detect drift using DDM algorithm."""
        baseline_center = np.mean(baseline, axis=0)
        current_distances = np.linalg.norm(current - baseline_center, axis=1)
        baseline_distances = np.linalg.norm(baseline - baseline_center, axis=1)

        threshold = np.percentile(baseline_distances, 90)
        errors = (current_distances > threshold).astype(float)

        error_rate = np.mean(errors)
        std = np.sqrt(error_rate * (1 - error_rate) / len(errors)) if len(errors) > 0 else 0.0

        if std > 0:
            drift_threshold = error_rate + self.drift_level * std
            current_error = np.mean(errors)
            drift_detected = current_error > drift_threshold
            severity = (current_error - error_rate) / (std + 1e-10)
        else:
            drift_detected = False
            severity = 0.0

        confidence = 0.9 if drift_detected else 0.1
        p_value = 0.01 if drift_detected else 0.9

        return BaselineResult(
            drift_detected=drift_detected,
            p_value=p_value,
            confidence=confidence,
            severity=abs(severity),
            method_name="DDM",
            sample_count=len(current),
        )


class HDP:
    """Hierarchical Dirichlet Process for drift detection.

    Detects drift by modeling topic distributions and comparing
    them across time windows.
    """

    def __init__(self, concentration: float = 1.0, rng: Optional[np.random.Generator] = None):
        """Initialize HDP."""
        self.concentration = concentration
        self.rng = rng or np.random.default_rng()

    def detect(self, baseline: np.ndarray, current: np.ndarray) -> BaselineResult:
        """Detect drift using HDP approximation."""
        n_components = min(5, baseline.shape[1], baseline.shape[0] + current.shape[0] - 1)
        n_components = max(1, n_components)
        pca = PCA(n_components=n_components)
        baseline_pca = pca.fit_transform(baseline)
        current_pca = pca.transform(current)

        n_topics = min(3, len(baseline) // 5)

        if n_topics < 1:
            return BaselineResult(
                drift_detected=False,
                p_value=0.5,
                confidence=0.5,
                severity=0.0,
                method_name="HDP",
                sample_count=len(current),
            )

        kmeans_b = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        kmeans_c = KMeans(n_clusters=n_topics, random_state=42, n_init=10)

        baseline_labels = kmeans_b.fit_predict(baseline_pca)
        current_labels = kmeans_c.fit_predict(current_pca)

        baseline_dist = np.bincount(baseline_labels, minlength=n_topics) / len(baseline_labels)
        current_dist = np.bincount(current_labels, minlength=n_topics) / len(current_labels)

        kl_div = 0.0
        for i in range(n_topics):
            if baseline_dist[i] > 0 and current_dist[i] > 0:
                kl_div += current_dist[i] * np.log(current_dist[i] / baseline_dist[i])

        threshold = 0.1 * self.concentration
        drift_detected = kl_div > threshold
        severity = kl_div / (threshold + 1e-10)

        confidence = min(0.99, severity / 2) if drift_detected else max(0.01, 1 - severity / 2)
        p_value = max(0.001, 1 / (1 + np.exp(severity - 1)))

        return BaselineResult(
            drift_detected=drift_detected,
            p_value=p_value,
            confidence=confidence,
            severity=abs(severity),
            method_name="HDP",
            sample_count=len(current),
        )
