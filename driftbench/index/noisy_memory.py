"""Composes the three noise channels into a single memory-store facade — the
main reusable artifact of the harness: one object another group can point at
their own agent stack to get a drift-under-infra readout, per
[INSIGHT: intro_contributions] (the harness as its own headline deliverable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from driftbench.index.approx_index import ApproxIndex, ApproxIndexConfig
from driftbench.index.eviction import BudgetedMemory, EvictionConfig
from driftbench.index.shadow_index import ShadowIndex, ShadowIndexConfig


@dataclass
class RetrievalReport:
    """Per-query instrumentation — the three checkable quantities, reported
    together so a single retrieval call yields all applicable diagnostics."""

    retrieved_ids: list[int]
    staleness: int  # systematic channel: pending-write count at query time
    recall_at_k: float  # stochastic channel: approx vs. exact overlap
    retention_rate: float | None  # selection channel: vs. oracle, if tracked


class NoisyMemoryStore:
    """A single agent-memory store wired to expose stochastic, systematic,
    and selection noise as independently controllable, independently
    measurable channels."""

    def __init__(
        self,
        dim: int,
        stochastic: ApproxIndexConfig,
        systematic: ShadowIndexConfig,
        selection: EvictionConfig | None = None,
        oracle: "NoisyMemoryStore | None" = None,
    ):
        self.dim = dim
        self._approx = ApproxIndex(dim, stochastic)
        self._shadow = ShadowIndex(dim, systematic)
        self._budget = BudgetedMemory(selection or EvictionConfig())
        self._oracle = oracle  # unbounded-memory reference run, for retention_rate
        self._next_id = 0

    def write(self, vector: np.ndarray, importance: float = 0.0) -> int:
        item_id = self._next_id
        self._next_id += 1
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        self._approx.add(vector.reshape(1, -1), [item_id])
        self._shadow.write(vector, item_id)
        self._budget.add(item_id, vector, importance=importance)
        return item_id

    def retrieve(self, query: np.ndarray, k: int) -> RetrievalReport:
        # Selection (budget) gates what's visible at all; among the budgeted
        # ids, staleness/recall are read off the approx/shadow instrumentation.
        visible_ids = self._budget.ids()
        candidate_ids = self._shadow.search(query, k)
        retrieved_ids = [i for i in candidate_ids if i in visible_ids][:k]
        for i in retrieved_ids:
            self._budget.touch(i)

        retention = None
        if self._oracle is not None:
            retention = self._budget.retention_rate(self._oracle._budget)

        return RetrievalReport(
            retrieved_ids=retrieved_ids,
            staleness=self._shadow.staleness_at_query(),
            recall_at_k=self._approx.recall_at_k(query, k),
            retention_rate=retention,
        )

    def flush(self) -> None:
        """Force any pending writes into the searchable index — call at
        episode boundaries per the testbed's protocol, or for the oracle run."""
        self._shadow.force_rebuild()
