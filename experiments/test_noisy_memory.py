import numpy as np

from driftbench.index.approx_index import ApproxIndexConfig
from driftbench.index.eviction import EvictionConfig
from driftbench.index.noisy_memory import NoisyMemoryStore
from driftbench.index.shadow_index import ShadowIndexConfig


def test_baseline_condition_is_exact_and_fresh():
    dim = 8
    store = NoisyMemoryStore(
        dim=dim,
        stochastic=ApproxIndexConfig(kind="flat"),
        systematic=ShadowIndexConfig(rebuild_cadence=1),
    )
    vec = np.ones(dim, dtype=np.float32)
    store.write(vec)

    report = store.retrieve(vec, k=1)
    assert report.staleness == 0
    assert report.recall_at_k == 1.0
    assert report.retrieved_ids == [0]


def test_stale_condition_reports_nonzero_staleness_and_hides_write():
    dim = 8
    store = NoisyMemoryStore(
        dim=dim,
        stochastic=ApproxIndexConfig(kind="flat"),
        systematic=ShadowIndexConfig(rebuild_cadence=10),
    )
    vec = np.ones(dim, dtype=np.float32)
    store.write(vec)

    report = store.retrieve(vec, k=1)
    assert report.staleness == 1
    assert report.retrieved_ids == []  # not yet merged into the searchable index


def test_selection_budget_gates_visibility_independent_of_staleness():
    dim = 8
    store = NoisyMemoryStore(
        dim=dim,
        stochastic=ApproxIndexConfig(kind="flat"),
        systematic=ShadowIndexConfig(rebuild_cadence=1),  # fresh: no staleness
        selection=EvictionConfig(budget=1, policy="fifo"),
    )
    vec = np.ones(dim, dtype=np.float32)
    store.write(vec)
    store.write(vec)  # evicts id=0 from the budget, even though it's fresh in the index

    report = store.retrieve(vec, k=2)
    assert report.staleness == 0  # systematic channel unaffected
    assert report.retrieved_ids == [1]  # only the surviving budgeted id is visible


def test_flush_makes_pending_writes_searchable():
    dim = 8
    store = NoisyMemoryStore(
        dim=dim,
        stochastic=ApproxIndexConfig(kind="flat"),
        systematic=ShadowIndexConfig(rebuild_cadence=100),
    )
    vec = np.ones(dim, dtype=np.float32)
    store.write(vec)
    assert store.retrieve(vec, k=1).retrieved_ids == []

    store.flush()
    assert store.retrieve(vec, k=1).retrieved_ids == [0]


def test_retrieval_routes_through_the_approximate_index():
    """Regression guard for the stochastic channel being on the retrieval path.

    Under an approximate (HNSW) index, the retrieved ids must be exactly what
    the approximate index returns (budget unbounded here). On the previous
    implementation retrieval came from an exact shadow index, so the stochastic
    level could not affect what was retrieved; that version fails this test.
    """
    dim = 8
    store = NoisyMemoryStore(
        dim=dim,
        stochastic=ApproxIndexConfig(kind="hnsw", hnsw_m=8, ef_search=8),
        systematic=ShadowIndexConfig(rebuild_cadence=1),
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        store.write(rng.standard_normal(dim).astype(np.float32))
    q = rng.standard_normal(dim).astype(np.float32)
    report = store.retrieve(q, k=3)
    assert report.retrieved_ids == store._approx.search(q, k=3)[:3]
