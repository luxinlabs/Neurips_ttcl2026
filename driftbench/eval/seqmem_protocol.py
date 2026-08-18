"""Evaluation protocol adopted from SeqMem-Eval (arXiv 2605.15384): separates
operational proxy metrics (task success, loss) from actual memory retention
(free-form recall with supporting context removed), checkpointed across
episodes rather than scored only at the end state.

This is the "recall, not loss" discipline referenced throughout PAPER_PLAN.md:
a memory method can look like it's working on a proxy metric while retention
is actually zero (per the "Beyond Perplexity" critique, arXiv 2607.00368,
surfaced in earlier research on this paper). Every checkpoint records both,
so the gap between them is directly reportable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Checkpoint:
    episode: int
    proxy_success: float  # e.g. downstream task success rate, in [0, 1]
    recall_score: float  # free-form recall accuracy w/ context removed, in [0, 1]
    forgetting: float = 0.0  # drop in recall_score on earlier-episode facts, in [0, 1]


@dataclass
class SeqMemTrace:
    """A run's full checkpoint history for one (condition, task instance) cell."""

    condition_name: str
    task_instance_id: str
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def add(self, checkpoint: Checkpoint) -> None:
        self.checkpoints.append(checkpoint)

    def proxy_recall_gap(self, episode: int) -> float:
        """The core validity check: proxy metrics improving while recall
        stays flat/zero is the failure mode this protocol exists to catch."""
        cp = self._at(episode)
        return cp.proxy_success - cp.recall_score

    def max_proxy_recall_gap(self) -> float:
        if not self.checkpoints:
            return 0.0
        return max(cp.proxy_success - cp.recall_score for cp in self.checkpoints)

    def forgetting_trajectory(self) -> list[tuple[int, float]]:
        return [(cp.episode, cp.forgetting) for cp in self.checkpoints]

    def _at(self, episode: int) -> Checkpoint:
        for cp in self.checkpoints:
            if cp.episode == episode:
                return cp
        raise KeyError(f"no checkpoint recorded for episode {episode}")
