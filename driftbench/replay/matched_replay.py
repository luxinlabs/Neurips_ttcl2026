"""Matched-replay orchestration: run the SAME starting task instance across
every grid condition, so infra is the sole source of divergence in what
follows (per [INSIGHT: thesis_statement] and [INSIGHT: counterargument_and_rebuttal]).

Agent/testbed integration is intentionally injected via `episode_runner`
rather than hard-coded here: base model choice and testbed adapters
(AgentOdyssey / Evo-Memory) are still open per PAPER_PLAN.md. This module
owns the part that's already locked — grid iteration, matched-instance
bookkeeping, and result collection — so it's fully testable now with a stub
runner, and the real agent loop drops in later without changing this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from driftbench.config import ConditionSpec, ExperimentGrid
from driftbench.eval.seqmem_protocol import SeqMemTrace
from driftbench.index.noisy_memory import NoisyMemoryStore


class TaskInstance(Protocol):
    """Opaque handle to a matched starting state — a seed/episode id in
    AgentOdyssey or Evo-Memory terms. driftbench never inspects its contents;
    it only guarantees the same instance is (re)used across every condition."""

    instance_id: str


EpisodeRunner = Callable[[TaskInstance, ConditionSpec, NoisyMemoryStore], SeqMemTrace]


@dataclass
class ReplayResult:
    condition_name: str
    instance_id: str
    trace: SeqMemTrace


def run_matched_replay(
    instances: list[TaskInstance],
    grid: ExperimentGrid,
    episode_runner: EpisodeRunner,
    dim: int,
    include_selection: bool = True,
) -> list[ReplayResult]:
    """For every task instance, replay it once per grid condition (9 factorial
    cells, plus Selection's one-at-a-time conditions if `include_selection`).
    Each (instance, condition) pair gets a freshly constructed NoisyMemoryStore
    — trajectories are expected to diverge after the first retrieval; only the
    starting instance is held fixed, per the matched-replay design.
    """
    results: list[ReplayResult] = []
    conditions = list(grid.factorial_cells)
    if include_selection:
        conditions += list(grid.selection_conditions)

    for instance in instances:
        for condition in conditions:
            store = NoisyMemoryStore(
                dim=dim,
                stochastic=condition.stochastic,
                systematic=condition.systematic,
                selection=condition.selection,
            )
            trace = episode_runner(instance, condition, store)
            results.append(
                ReplayResult(
                    condition_name=condition.name,
                    instance_id=instance.instance_id,
                    trace=trace,
                )
            )
    return results
