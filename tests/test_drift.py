import numpy as np
import pytest

from driftbench.metrics.drift import DriftTracker, entropy, kl_divergence


def test_entropy_of_uniform_distribution_is_maximal():
    uniform = np.ones(4) / 4
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    assert entropy(uniform) > entropy(peaked)


def test_kl_divergence_is_zero_for_identical_distributions():
    p = np.array([0.5, 0.3, 0.2])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_is_nonzero_for_different_distributions():
    p = np.array([0.9, 0.05, 0.05])
    q = np.array([0.05, 0.05, 0.9])
    assert kl_divergence(p, q) > 0.5


def test_drift_tracker_reports_zero_drift_at_reference_free_baseline():
    tracker = DriftTracker(reference_episode=0)
    dist = np.array([0.5, 0.5])
    tracker.record(0, dist)
    tracker.record(1, dist)  # identical distribution: no drift
    assert tracker.drift_at(1) == pytest.approx(0.0, abs=1e-9)


def test_drift_tracker_detects_systematic_style_directional_drift():
    """Sanity check for the paper's core mechanism: a distribution that moves
    steadily away from the reference (systematic-style compounding) should
    show monotonically increasing drift, unlike noise that doesn't compound."""
    tracker = DriftTracker(reference_episode=0)
    tracker.record(0, np.array([0.9, 0.1]))
    tracker.record(1, np.array([0.7, 0.3]))
    tracker.record(2, np.array([0.5, 0.5]))
    tracker.record(3, np.array([0.3, 0.7]))

    trajectory = tracker.drift_trajectory()
    values = [trajectory[e] for e in sorted(trajectory)]
    assert values == sorted(values)  # monotonically increasing drift
