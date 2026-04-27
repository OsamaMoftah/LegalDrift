"""Drift detection combining multiple statistical methods.

Multi-method drift detection with Fisher's method for combining p-values.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection analysis."""

    drift_detected: bool
    p_value: float
    confidence: float
    severity: float
    effect_size: float
    threshold: float
    tests: Dict[str, Dict[str, float]]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "drift_detected": self.drift_detected,
            "p_value": self.p_value,
            "confidence": self.confidence,
            "severity": self.severity,
            "effect_size": self.effect_size,
            "threshold": self.threshold,
            "tests": self.tests,
        }


class DriftDetector:
    """Multi-method statistical drift detector.

    Combines KS test, Mann-Whitney, MMD, and Energy distance
    with Fisher's method for robust drift detection.
    """

    def __init__(
        self, threshold: float = 0.05, n_permutations: int = 200, rng: np.random.Generator = None
    ):
        """Initialize detector.

        Args:
            threshold: P-value threshold for drift detection.
            n_permutations: Number of permutations for statistical tests.
            rng: Random number generator for reproducibility.
        """
        self.threshold = threshold
        self.n_permutations = n_permutations
        self.rng = rng or np.random.default_rng()

    def detect(self, baseline: np.ndarray, current: np.ndarray) -> DriftResult:
        """Detect drift between baseline and current distributions.

        Args:
            baseline: Baseline embeddings, shape (n_samples, n_features).
            current: Current embeddings to compare, shape (m_samples, n_features).

        Returns:
            DriftResult with detection status and statistics.
        """
        logger.info("Detecting drift: baseline=%s, current=%s", baseline.shape, current.shape)

        baseline_pca, current_pca = self._reduce_dimensions(baseline, current)

        ks_stat, ks_p = self._ks_test(baseline_pca, current_pca)
        mw_stat, mw_p = self._mann_whitney(baseline_pca, current_pca)
        mmd_p = self._permutation_test(baseline, current, self._mmd)
        energy_p = self._permutation_test(baseline, current, self._energy_distance)

        p_values = [ks_p, mw_p, mmd_p, energy_p]
        combined_stat, combined_p = self._fisher_combine(p_values)

        severity = self._wasserstein_severity(baseline_pca, current_pca)
        confidence = max(0.001, min(0.999, 1.0 - combined_p))
        effect_size = self._cohens_d(baseline_pca, current_pca)

        drift_detected = bool(combined_p < self.threshold)

        tests = {
            "ks_test": {"statistic": float(ks_stat), "p_value": float(ks_p)},
            "mannwhitney": {"statistic": float(mw_stat), "p_value": float(mw_p)},
            "mmd": {"p_value": float(mmd_p)},
            "energy": {"p_value": float(energy_p)},
        }

        result = DriftResult(
            drift_detected=drift_detected,
            p_value=float(combined_p),
            confidence=float(confidence),
            severity=float(severity),
            effect_size=float(effect_size),
            threshold=self.threshold,
            tests=tests,
        )

        logger.info(
            "Drift detection complete: drift=%s, p=%.4f, severity=%.3f",
            drift_detected,
            combined_p,
            severity,
        )
        return result

    def _reduce_dimensions(
        self, X: np.ndarray, Y: np.ndarray, n_components: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce dimensions using PCA for faster testing."""
        n_comp = min(n_components, X.shape[1], Y.shape[1], X.shape[0] + Y.shape[0] - 1)
        n_comp = max(1, n_comp)
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        Y_pca = pca.transform(Y)
        return X_pca, Y_pca

    def _ks_test(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Multivariate Kolmogorov-Smirnov test across all dimensions.

        Aggregates p-values across dimensions using Fisher's method,
        then returns the mean KS statistic and combined p-value.
        """
        p_values = []
        stats_list = []
        n_dims = X.shape[1]
        for dim in range(n_dims):
            stat, p = stats.ks_2samp(X[:, dim], Y[:, dim])
            p_values.append(p)
            stats_list.append(stat)
        # Combine p-values across dimensions
        _, combined_p = self._fisher_combine(p_values)
        mean_stat = float(np.mean(stats_list))
        return mean_stat, combined_p

    def _mann_whitney(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Multivariate Mann-Whitney U test across all dimensions.

        Aggregates p-values across dimensions using Fisher's method,
        then returns the mean U-statistic and combined p-value.
        """
        p_values = []
        stats_list = []
        n_dims = X.shape[1]
        for dim in range(n_dims):
            stat, p = stats.mannwhitneyu(X[:, dim], Y[:, dim], alternative="two-sided")
            p_values.append(p)
            stats_list.append(stat)
        _, combined_p = self._fisher_combine(p_values)
        mean_stat = float(np.mean(stats_list))
        return mean_stat, combined_p

    def _permutation_test(
        self, X: np.ndarray, Y: np.ndarray, metric_fn, n_perms: int = None
    ) -> float:
        """Permutation test for any metric function."""
        n_perms = n_perms or self.n_permutations
        observed = metric_fn(X, Y)
        combined = np.vstack([X, Y])
        n_x = X.shape[0]

        null_stats = []
        for _ in range(n_perms):
            shuffled = self.rng.permutation(combined)
            X_perm, Y_perm = shuffled[:n_x], shuffled[n_x:]
            null_stats.append(metric_fn(X_perm, Y_perm))

        return float((np.sum(np.array(null_stats) >= observed) + 1) / (n_perms + 1))

    def _mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """Maximum Mean Discrepancy."""
        XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
        YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :]) ** 2, axis=2))
        XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))
        return float(XX.mean() + YY.mean() - 2 * XY.mean())

    def _energy_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Energy distance between distributions."""
        XX = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        YY = np.linalg.norm(Y[:, None] - Y[None, :], axis=2)
        XY = np.linalg.norm(X[:, None] - Y[None, :], axis=2)
        return float(2 * XY.mean() - XX.mean() - YY.mean())

    def _wasserstein_severity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Wasserstein distance as drift severity measure."""
        try:
            from scipy.stats import wasserstein_distance

            return wasserstein_distance(X[:, 0], Y[:, 0])
        except ImportError:
            return float(abs(np.mean(X) - np.mean(Y)))

    def _cohens_d(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Cohen's d effect size."""
        pooled_std = np.sqrt(
            ((len(X) - 1) * np.var(X, ddof=1) + (len(Y) - 1) * np.var(Y, ddof=1))
            / (len(X) + len(Y) - 2)
        )
        if pooled_std == 0 or np.isnan(pooled_std):
            return 0.0
        return float((np.mean(X) - np.mean(Y)) / pooled_std)

    def _fisher_combine(self, p_values: List[float]) -> Tuple[float, float]:
        """Fisher's method for combining p-values."""
        valid_p = [p for p in p_values if p > 0]
        if not valid_p:
            return float("inf"), 0.0

        chi2_stat = -2 * sum(math.log(p) for p in valid_p)
        df = 2 * len(valid_p)
        p_combined = 1 - stats.chi2.cdf(chi2_stat, df)
        return float(chi2_stat), float(p_combined)
