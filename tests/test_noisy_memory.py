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
