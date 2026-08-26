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
from driftbench.index.eviction import EvictionConfig
from driftbench.index.noisy_memory import NoisyMemoryStore


class TaskInstance(Protocol):
    """Opaque handle to a matched starting state — a seed/episode id in
    AgentOdyssey or Evo-Memory terms. driftbench never inspects its contents;
    it only guarantees the same instance is (re)used across every condition."""

    instance_id: str


EpisodeRunner = Callable[[TaskInstance, ConditionSpec, NoisyMemoryStore, int], SeqMemTrace]


@dataclass
class ReplayResult:
    condition_name: str
    instance_id: str
    replicate: int
    trace: SeqMemTrace


def run_matched_replay(
    instances: list[TaskInstance],
    grid: ExperimentGrid,
    episode_runner: EpisodeRunner,
    dim: int,
    include_selection: bool = True,
    n_samples_per_cell: int = 1,
    measure_retention: bool = False,
) -> list[ReplayResult]:
    """For every task instance, replay it once per grid condition (9 factorial
    cells, plus Selection's one-at-a-time conditions if `include_selection`).
    Each (instance, condition) pair gets a freshly constructed NoisyMemoryStore
    — trajectories are expected to diverge after the first retrieval; only the
    starting instance is held fixed, per the matched-replay design.

    `n_samples_per_cell` controls how many independent replicates are run per
    (instance, condition) pair. Default 1 preserves prior behavior. Setting
    this above 1 is what makes sampling variance (the generation model's own
    stochastic decoding) separately identifiable from between-instance
    heterogeneity in the mixed-effects analysis: with exactly one replicate
    per cell there is only one observation to estimate both quantities from,
    so they cannot be told apart. `episode_runner` receives the 0-indexed
    replicate number as its last argument so callers can, e.g., vary a
    generation seed deterministically per replicate.
    """
    if n_samples_per_cell < 1:
        raise ValueError("n_samples_per_cell must be >= 1")

    results: list[ReplayResult] = []
    conditions = list(grid.factorial_cells)
    if include_selection:
        conditions += list(grid.selection_conditions)

    for instance in instances:
        # Selection channel's retention metric compares a budgeted memory to an
        # unbounded-memory ("oracle") run of the SAME task instance. When
        # requested, run that oracle once per instance (baseline infra, no
        # eviction) and thread it into every condition's store so retrieve()
        # can report retention_rate. NOTE: this compares against the oracle's
        # final contents; per-query lockstep time-alignment is a refinement
        # left as a TODO. Off by default: it costs one extra rollout/instance.
        oracle_store: NoisyMemoryStore | None = None
        if measure_retention and include_selection:
            oracle_store = NoisyMemoryStore(
                dim=dim,
                stochastic=grid.baseline.stochastic,
                systematic=grid.baseline.systematic,
                selection=EvictionConfig(budget=None),  # unbounded
            )
            episode_runner(instance, grid.baseline, oracle_store, 0)

        for condition in conditions:
            for replicate in range(n_samples_per_cell):
                store = NoisyMemoryStore(
                    dim=dim,
                    stochastic=condition.stochastic,
                    systematic=condition.systematic,
                    selection=condition.selection,
                    oracle=oracle_store,
                )
                trace = episode_runner(instance, condition, store, replicate)
                results.append(
                    ReplayResult(
                        condition_name=condition.name,
                        instance_id=instance.instance_id,
                        replicate=replicate,
                        trace=trace,
                    )
                )
    return results
