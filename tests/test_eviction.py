import numpy as np

from driftbench.index.eviction import BudgetedMemory, EvictionConfig


def _vec(dim=4):
    return np.ones(dim, dtype=np.float32)


def test_unbounded_memory_never_evicts():
    mem = BudgetedMemory(EvictionConfig(budget=None))
    for i in range(100):
        mem.add(i, _vec())
    assert len(mem.ids()) == 100


def test_fifo_evicts_oldest_first():
    mem = BudgetedMemory(EvictionConfig(budget=3, policy="fifo"))
    for i in range(5):
        mem.add(i, _vec())
    assert mem.ids() == {2, 3, 4}


def test_lru_keeps_recently_touched_items():
    mem = BudgetedMemory(EvictionConfig(budget=3, policy="lru"))
    mem.add(0, _vec())
    mem.add(1, _vec())
    mem.add(2, _vec())
    mem.touch(0)  # 0 is now most-recently-used, despite being oldest inserted
    mem.add(3, _vec())  # forces an eviction
    assert 0 in mem.ids()  # protected by the touch
    assert mem.ids() == {0, 2, 3}


def test_importance_evicts_lowest_importance_first():
    mem = BudgetedMemory(EvictionConfig(budget=2, policy="importance"))
    mem.add(0, _vec(), importance=0.9)
    mem.add(1, _vec(), importance=0.1)
    mem.add(2, _vec(), importance=0.5)  # triggers eviction of id=1 (lowest importance)
    assert mem.ids() == {0, 2}


def test_retention_rate_against_oracle():
    oracle = BudgetedMemory(EvictionConfig(budget=None))
    budgeted = BudgetedMemory(EvictionConfig(budget=2, policy="fifo"))

    for i in range(4):
        oracle.add(i, _vec())
        budgeted.add(i, _vec())

    # oracle retains {0,1,2,3}; budgeted (fifo, budget=2) retains {2,3}
    assert budgeted.retention_rate(oracle) == 0.5
