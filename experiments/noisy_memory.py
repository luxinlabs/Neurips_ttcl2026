"""Composes the three noise channels into a single memory-store facade — the
main reusable artifact of the harness: one object another group can point at
their own agent stack to get a drift-under-infra readout, per
[INSIGHT: intro_contributions] (the harness as its own headline deliverable).

Retrieval routing (fixed): a write first lands in a staleness buffer and only
becomes searchable when the rebuild cadence flushes it into the *approximate*
index. Retrieval then runs approximate search over that persisted set and is
gated by the selection budget. This is what puts all three channels on the
actual retrieval path at once:

  * systematic (staleness): a pending write is invisible until the next rebuild;
  * stochastic (approximate search): the persisted set is searched approximately,
    so the stochastic level actually determines which neighbours come back
    (previously the approximate index was measured but never queried, so the
    stochastic channel could not causally affect behaviour);
  * selection (budget): eviction gates which persisted ids are visible at all.

ApproxIndex keeps an internal exact reference over exactly the vectors it holds,
so recall@k is computed against the same persisted set the agent actually sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from driftbench.index.approx_index import ApproxIndex, ApproxIndexConfig
from driftbench.index.eviction import BudgetedMemory, EvictionConfig
from driftbench.index.shadow_index import ShadowIndexConfig


@dataclass
class RetrievalReport:
    """Per-query instrumentation — the three checkable quantities, reported
    together so a single retrieval call yields all applicable diagnostics."""

    retrieved_ids: list[int]
    retrieved_payloads: list[str]  # the actual memory text/notes, id-aligned
    staleness: int  # systematic channel: pending-write count at query time
    recall_at_k: float  # stochastic channel: approx vs. exact overlap
    retention_rate: float | None  # selection channel: vs. oracle, if tracked


class NoisyMemoryStore:
    """A single agent-memory store wired to expose stochastic, systematic,
    and selection noise as independently controllable, independently
    measurable channels — all three on the actual retrieval path."""

    def __init__(
        self,
        dim: int,
        stochastic: ApproxIndexConfig,
        systematic: ShadowIndexConfig,
        selection: EvictionConfig | None = None,
        oracle: "NoisyMemoryStore | None" = None,
    ):
        self.dim = dim
        # Persisted, searchable index (approximate). Also holds an internal
        # exact reference over the same vectors, so recall@k is ground-truthed
        # against exactly what is searchable.
        self._approx = ApproxIndex(dim, stochastic)
        # Systematic channel: a write is buffered here and only flushed into
        # _approx every `rebuild_cadence` writes (K=1 => always fresh).
        self._rebuild_cadence = max(1, systematic.rebuild_cadence)
        self._pending: list[tuple[int, np.ndarray]] = []
        self._writes_since_rebuild = 0
        # Selection channel: budgeted visibility over all writes.
        self._budget = BudgetedMemory(selection or EvictionConfig())
        self._oracle = oracle  # unbounded-memory reference run, for retention_rate
        self._next_id = 0
        self._payloads: dict[int, str] = {}

    def write(self, vector: np.ndarray, payload: str = "", importance: float = 0.0) -> int:
        item_id = self._next_id
        self._next_id += 1
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        # Buffer first (staleness); persist into the searchable index on cadence.
        self._pending.append((item_id, vector))
        self._writes_since_rebuild += 1
        if self._writes_since_rebuild >= self._rebuild_cadence:
            self._rebuild()
        # The budget tracks every write immediately — eviction is a visibility
        # gate independent of index freshness.
        self._budget.add(item_id, vector, importance=importance)
        self._payloads[item_id] = payload
        return item_id

    def _rebuild(self) -> None:
        if not self._pending:
            self._writes_since_rebuild = 0
            return
        vecs = np.stack([v for _, v in self._pending])
        ids = [i for i, _ in self._pending]
        self._approx.add(vecs, ids)
        self._pending.clear()
        self._writes_since_rebuild = 0

    def retrieve(self, query: np.ndarray, k: int) -> RetrievalReport:
        # Approximate search over the persisted set is the retrieval path;
        # the budget then gates which of those ids are visible.
        visible_ids = self._budget.ids()
        candidate_ids = self._approx.search(query, k)
        retrieved_ids = [i for i in candidate_ids if i in visible_ids][:k]
        for i in retrieved_ids:
            self._budget.touch(i)

        retention = None
        if self._oracle is not None:
            retention = self._budget.retention_rate(self._oracle._budget)

        return RetrievalReport(
            retrieved_ids=retrieved_ids,
            retrieved_payloads=[self._payloads.get(i, "") for i in retrieved_ids],
            staleness=len(self._pending),  # systematic: writes not yet searchable
            recall_at_k=self._approx.recall_at_k(query, k),
            retention_rate=retention,
        )

    def flush(self) -> None:
        """Force any pending writes into the searchable index — call at
        episode boundaries per the testbed's protocol, or for the oracle run."""
        self._rebuild()
