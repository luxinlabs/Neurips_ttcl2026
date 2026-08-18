"""Selection-noise channel: eviction policy under a fixed memory budget.

SECONDARY / APPENDIX-TIER per PAPER_PLAN.md [INSIGHT: taxonomy_operationalization]
and [INSIGHT: ablation_design] — kept one-at-a-time against the shared baseline,
not crossed into the headline stochastic x systematic grid. Its metric
(necessary-fact retention rate) is a novel proposal, not adopted from prior
work, unlike recall@k and the staleness definition used by the other two
channels; this is flagged deliberately rather than presented as equally
established.

Checkable quantity: retention rate against an unbounded-memory oracle run of
the SAME task instance — i.e. does the budgeted memory still hold what an
unbounded memory would hold, at each query point? This avoids depending on
environment-exposed ground truth about which facts are "necessary."
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import numpy as np

Policy = Literal["fifo", "lru", "importance"]


@dataclass
class EvictionConfig:
    budget: int | None = None  # None = unbounded (the oracle condition)
    policy: Policy = "fifo"


class BudgetedMemory:
    def __init__(self, config: EvictionConfig):
        self.config = config
        # id -> vector, insertion-ordered; also doubles as LRU order via
        # move-to-end on access.
        self._store: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._importance: dict[int, float] = {}

    def add(self, item_id: int, vector: np.ndarray, importance: float = 0.0) -> None:
        self._store[item_id] = np.ascontiguousarray(vector, dtype=np.float32)
        self._importance[item_id] = importance
        self._evict_if_needed()

    def touch(self, item_id: int) -> None:
        """Record an access, for LRU policy bookkeeping."""
        if item_id in self._store:
            self._store.move_to_end(item_id)

    def _evict_if_needed(self) -> None:
        budget = self.config.budget
        if budget is None:
            return
        while len(self._store) > budget:
            evict_id = self._select_eviction_candidate()
            del self._store[evict_id]
            self._importance.pop(evict_id, None)

    def _select_eviction_candidate(self) -> int:
        if self.config.policy in ("fifo", "lru"):
            # OrderedDict: oldest-inserted (fifo) / least-recently-touched
            # (lru, via move_to_end on access) is the first key.
            return next(iter(self._store))
        if self.config.policy == "importance":
            return min(self._importance, key=self._importance.get)
        raise ValueError(f"unknown eviction policy: {self.config.policy!r}")

    def ids(self) -> set[int]:
        return set(self._store.keys())

    def retention_rate(self, oracle: "BudgetedMemory") -> float:
        """Fraction of the oracle's (unbounded-memory) current contents that
        this budgeted memory still holds, at this point in the replay."""
        oracle_ids = oracle.ids()
        if not oracle_ids:
            return 1.0
        return len(self.ids() & oracle_ids) / len(oracle_ids)
