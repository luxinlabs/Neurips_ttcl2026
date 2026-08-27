import numpy as np

from driftbench.index.approx_index import ApproxIndex, ApproxIndexConfig


def _random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_flat_index_has_perfect_recall():
    dim = 16
    vecs = _random_vectors(200, dim)
    index = ApproxIndex(dim, ApproxIndexConfig(kind="flat"))
    index.add(vecs, list(range(len(vecs))))

    query = vecs[0]
    assert index.recall_at_k(query, k=10) == 1.0


def test_aggressive_hnsw_recall_is_at_most_flat_recall():
    dim = 16
    vecs = _random_vectors(500, dim)

    flat = ApproxIndex(dim, ApproxIndexConfig(kind="flat"))
    flat.add(vecs, list(range(len(vecs))))

    aggressive = ApproxIndex(
        dim, ApproxIndexConfig(kind="hnsw", hnsw_m=4, ef_search=1, ef_construction=8)
    )
    aggressive.add(vecs, list(range(len(vecs))))

    query = vecs[123]
    flat_recall = flat.recall_at_k(query, k=10)
    aggressive_recall = aggressive.recall_at_k(query, k=10)

    assert flat_recall == 1.0
    assert aggressive_recall <= flat_recall


def test_search_returns_k_ids():
    dim = 8
    vecs = _random_vectors(50, dim)
    index = ApproxIndex(dim, ApproxIndexConfig(kind="flat"))
    index.add(vecs, list(range(len(vecs))))

    results = index.search(vecs[0], k=5)
    assert len(results) == 5
    assert 0 in results  # nearest neighbor of itself


def test_empty_index_search_is_safe():
    index = ApproxIndex(8, ApproxIndexConfig(kind="flat"))
    query = np.ones(8, dtype=np.float32)
    assert index.search(query, k=3) == []
    assert index.recall_at_k(query, k=3) == 1.0
