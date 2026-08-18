import pytest

from driftbench.eval.seqmem_protocol import Checkpoint, SeqMemTrace


def test_proxy_recall_gap_flags_the_beyond_perplexity_failure_mode():
    """Regression test for the exact pattern flagged in arXiv 2607.00368:
    proxy metric improving while recall stays at zero."""
    trace = SeqMemTrace(condition_name="stochastic=aggressive,systematic=stale", task_instance_id="t0")
    trace.add(Checkpoint(episode=0, proxy_success=0.9, recall_score=0.0))

    assert trace.proxy_recall_gap(0) == pytest.approx(0.9)
    assert trace.max_proxy_recall_gap() == pytest.approx(0.9)


def test_no_gap_when_proxy_and_recall_agree():
    trace = SeqMemTrace(condition_name="baseline", task_instance_id="t0")
    trace.add(Checkpoint(episode=0, proxy_success=0.8, recall_score=0.8))
    assert trace.proxy_recall_gap(0) == pytest.approx(0.0)


def test_forgetting_trajectory_tracks_across_episodes():
    trace = SeqMemTrace(condition_name="c", task_instance_id="t0")
    trace.add(Checkpoint(episode=0, proxy_success=0.8, recall_score=0.8, forgetting=0.0))
    trace.add(Checkpoint(episode=1, proxy_success=0.7, recall_score=0.6, forgetting=0.1))
    trace.add(Checkpoint(episode=2, proxy_success=0.6, recall_score=0.4, forgetting=0.3))

    trajectory = trace.forgetting_trajectory()
    assert trajectory == [(0, 0.0), (1, 0.1), (2, 0.3)]


def test_missing_checkpoint_raises():
    trace = SeqMemTrace(condition_name="c", task_instance_id="t0")
    with pytest.raises(KeyError):
        trace.proxy_recall_gap(5)
