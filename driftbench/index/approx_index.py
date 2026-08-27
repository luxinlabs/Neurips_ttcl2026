"""Stochastic-noise channel: approximate ANN search vs. exact search.

Checkable quantity (per PAPER_PLAN.md [INSIGHT: taxonomy_operationalization]):
recall@k of the approximate index against a same-vectors exact (flat) index.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass(frozen=True)
class ApproxIndexConfig:
    """One grid level of the stochastic channel.

    kind: "flat" (exact, the baseline level), "ivf", or "hnsw".
    nlist / nprobe: IVF parameters (ignored for flat/hnsw).
    ef_search / ef_construction / M: HNSW parameters (ignored for flat/ivf).
    """

    kind: str = "flat"
    nlist: int = 100
    nprobe: int = 8
    hnsw_m: int = 32
    ef_search: int = 16
    ef_construction: int = 40


class ApproxIndex:
    """Wraps a FAISS approximate index alongside an exact reference index
    built from the same vectors, so recall@k is always computable against
    ground truth rather than assumed from index parameters alone.
    """

    def __init__(self, dim: int, config: ApproxIndexConfig):
        self.dim = dim
        self.config = config
        self._exact = faiss.IndexFlatL2(dim)
        self._approx = self._build_approx_index(dim, config)
        self._ids: list[int] = []

    @staticmethod
    def _build_approx_index(dim: int, config: ApproxIndexConfig) -> faiss.Index:
        if config.kind == "flat":
            return faiss.IndexFlatL2(dim)
        if config.kind == "ivf":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, config.nlist)
            index.nprobe = config.nprobe
            return index
        if config.kind == "hnsw":
            index = faiss.IndexHNSWFlat(dim, config.hnsw_m)
            index.hnsw.efSearch = config.ef_search
            index.hnsw.efConstruction = config.ef_construction
            return index
        raise ValueError(f"unknown approx index kind: {config.kind!r}")

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if not self._approx.is_trained:
            self._approx.train(vectors)
        self._exact.add(vectors)
        self._approx.add(vectors)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[int]:
        """Return the k nearest item ids under the approximate index."""
        if not self._ids:
            # Nothing persisted yet (e.g. step 0, or a stale index whose writes
            # are all still pending). An untrained IVF index would raise here,
            # so short-circuit to an empty result.
            return []
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        _, idx = self._approx.search(query, k)
        return [self._ids[i] for i in idx[0] if i != -1]

    def recall_at_k(self, query: np.ndarray, k: int) -> float:
        """Fraction of the exact top-k that the approximate index also returns.
        This is the reported "dose" for the stochastic channel — 1.0 at the
        baseline (flat) level, dropping monotonically as approximation grows
        more aggressive.
        """
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if not self._ids:
            return 1.0  # nothing persisted yet: no approximation error to report
        _, exact_idx = self._exact.search(query, k)
        _, approx_idx = self._approx.search(query, k)
        exact_set = {i for i in exact_idx[0] if i != -1}
        approx_set = {i for i in approx_idx[0] if i != -1}
        if not exact_set:
            return 1.0
        return len(exact_set & approx_set) / len(exact_set)
