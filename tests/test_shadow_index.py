import numpy as np

from driftbench.index.shadow_index import ShadowIndex, ShadowIndexConfig


def test_fresh_index_k1_has_zero_staleness_after_each_write():
    dim = 8
    index = ShadowIndex(dim, ShadowIndexConfig(rebuild_cadence=1))
    vec = np.ones(dim, dtype=np.float32)

    index.write(vec, item_id=0)
    assert index.staleness_at_query() == 0  # merged immediately at K=1


def test_stale_index_hides_recent_writes_until_rebuild_cadence():
    dim = 8
    index = ShadowIndex(dim, ShadowIndexConfig(rebuild_cadence=5))
    vec = np.ones(dim, dtype=np.float32)

    for i in range(3):
        index.write(vec, item_id=i)

    # 3 writes, cadence 5: nothing merged yet, all 3 pending
    assert index.staleness_at_query() == 3
    assert index.search(vec, k=3) == []  # invisible: not yet in the live index


def test_rebuild_cadence_merges_pending_writes_and_resets_staleness():
    dim = 8
    index = ShadowIndex(dim, ShadowIndexConfig(rebuild_cadence=3))
    vec = np.ones(dim, dtype=np.float32)

    index.write(vec, item_id=0)
    index.write(vec, item_id=1)
    assert index.staleness_at_query() == 2

    index.write(vec, item_id=2)  # triggers rebuild at cadence=3
    assert index.staleness_at_query() == 0
    assert set(index.search(vec, k=3)) == {0, 1, 2}


def test_force_rebuild_flushes_pending_writes():
    dim = 8
    index = ShadowIndex(dim, ShadowIndexConfig(rebuild_cadence=100))
    vec = np.ones(dim, dtype=np.float32)

    index.write(vec, item_id=0)
    assert index.staleness_at_query() == 1

    index.force_rebuild()
    assert index.staleness_at_query() == 0
    assert index.search(vec, k=1) == [0]


def test_stale_fact_persists_after_being_superseded():
    """Regression test for the Introduction's Ward-3 hook mechanism: an
    updated fact written after the original isn't retrievable until rebuild,
    so search still returns the OLD id even though a corrected write exists."""
    dim = 4
    index = ShadowIndex(dim, ShadowIndexConfig(rebuild_cadence=10))
    old_fact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    updated_fact = np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32)  # near-identical query point

    index.write(old_fact, item_id="old_rule")
    index.force_rebuild()  # old fact is live

    index.write(updated_fact, item_id="new_rule")  # pending, not yet merged (cadence=10)

    results = index.search(updated_fact, k=1)
    assert results == ["old_rule"]  # stale: the correction is invisible
    assert index.staleness_at_query() == 1
