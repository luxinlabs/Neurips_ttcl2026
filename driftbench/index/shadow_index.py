"""Systematic-noise channel: index staleness under async/batched consolidation.

Checkable quantity (per PAPER_PLAN.md [INSIGHT: taxonomy_operationalization]):
pending-write count at query time, per the staleness definition in
Agent Memory Characterization (arXiv 2606.06448) — the number of prior writes
not yet persisted into the searchable ("live") index when a given query fires.

Mechanism: writes land in a pending buffer first. The buffer is merged into
the live index only every `rebuild_cadence` writes. Queries only ever see the
live index, so a fact written since the last rebuild is retrievable-invisible
until the next merge — this is what lets an agent keep confidently retrieving
and acting on a stale fact even after a corrected/updated one has been written.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np


@dataclass
class ShadowIndexConfig:
    rebuild_cadence: int = 1  # K=1 is the baseline (fresh / no staleness) level


class ShadowIndex:
    def __init__(self, dim: int, config: ShadowIndexConfig):
        self.dim = dim
        self.config = config
        self._live = faiss.IndexFlatL2(dim)
        self._live_ids: list[int] = []
        self._pending: list[tuple[int, np.ndarray]] = []
        self._writes_since_rebuild = 0

    def write(self, vector: np.ndarray, item_id: int) -> None:
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        self._pending.append((item_id, vector))
        self._writes_since_rebuild += 1
        if self._writes_since_rebuild >= self.config.rebuild_cadence:
            self._rebuild()

    def _rebuild(self) -> None:
        if not self._pending:
            self._writes_since_rebuild = 0
            return
        vecs = np.stack([v for _, v in self._pending])
        ids = [i for i, _ in self._pending]
        self._live.add(vecs)
        self._live_ids.extend(ids)
        self._pending.clear()
        self._writes_since_rebuild = 0

    def force_rebuild(self) -> None:
        """Explicit flush, e.g. at the end of an episode / for the oracle run."""
        self._rebuild()

    def staleness_at_query(self) -> int:
        """The checkable quantity: how many writes are sitting unindexed
        right now, i.e. invisible to the next search() call."""
        return len(self._pending)

    def search(self, query: np.ndarray, k: int) -> list[int]:
        if self._live.ntotal == 0:
            return []
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        _, idx = self._live.search(query, min(k, self._live.ntotal))
        return [self._live_ids[i] for i in idx[0] if i != -1]
