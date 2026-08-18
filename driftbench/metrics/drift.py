"""Drift detection metric, adapted from RDumb++ (arXiv 2601.15544): tracks
entropy / KL-divergence of the agent's output distribution across episodes
as a model-agnostic proxy for behavioral drift.

Model-agnostic by design: takes probability distributions (e.g. over
action/answer choices) as input rather than raw model internals, so it is
testable independent of the base-model choice (still open per
PAPER_PLAN.md [INSIGHT: sample_size_and_analysis]).
"""
from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + eps)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q): how much the current-episode distribution q has drifted
    from a reference (e.g. first-episode or pre-self-feedback) distribution p."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


class DriftTracker:
    """Accumulates per-episode output distributions and reports drift of each
    episode relative to a fixed reference episode (default: episode 0, i.e.
    pre-self-feedback behavior)."""

    def __init__(self, reference_episode: int = 0):
        self.reference_episode = reference_episode
        self._episodes: dict[int, np.ndarray] = {}

    def record(self, episode: int, distribution: np.ndarray) -> None:
        self._episodes[episode] = np.asarray(distribution, dtype=np.float64)

    def drift_at(self, episode: int) -> float:
        if self.reference_episode not in self._episodes:
            raise ValueError(
                f"reference episode {self.reference_episode} not recorded yet"
            )
        if episode not in self._episodes:
            raise ValueError(f"episode {episode} not recorded")
        ref = self._episodes[self.reference_episode]
        cur = self._episodes[episode]
        return kl_divergence(ref, cur)

    def drift_trajectory(self) -> dict[int, float]:
        return {ep: self.drift_at(ep) for ep in sorted(self._episodes) if ep != self.reference_episode}
