from dataclasses import dataclass
from pathlib import Path

from driftbench.config import ConditionSpec, build_grid
from driftbench.eval.seqmem_protocol import Checkpoint, SeqMemTrace
from driftbench.index.noisy_memory import NoisyMemoryStore
from driftbench.replay.matched_replay import run_matched_replay

GRID_PATH = Path(__file__).parent.parent / "configs" / "grid_default.yaml"


@dataclass
class FakeTaskInstance:
    instance_id: str


def _stub_runner(instance: FakeTaskInstance, condition: ConditionSpec, store: NoisyMemoryStore) -> SeqMemTrace:
    """A stand-in for the real agent loop (base model TBD). Just proves the
    orchestrator wires instance/condition/store together correctly and that
    the SAME instance_id is reused across every condition (matched-replay)."""
    trace = SeqMemTrace(condition_name=condition.name, task_instance_id=instance.instance_id)
    trace.add(Checkpoint(episode=0, proxy_success=0.5, recall_score=0.5))
    return trace


def test_matched_replay_reuses_the_same_instance_across_all_conditions():
    grid = build_grid(GRID_PATH)
    instances = [FakeTaskInstance("task-0"), FakeTaskInstance("task-1")]

    results = run_matched_replay(instances, grid, _stub_runner, dim=8)

    expected_conditions = len(grid.factorial_cells) + len(grid.selection_conditions)
    assert len(results) == len(instances) * expected_conditions

    for instance in instances:
        instance_results = [r for r in results if r.instance_id == instance.instance_id]
        seen_conditions = {r.condition_name for r in instance_results}
        expected_names = {c.name for c in grid.factorial_cells} | {
            c.name for c in grid.selection_conditions
        }
        assert seen_conditions == expected_names


def test_excluding_selection_conditions():
    grid = build_grid(GRID_PATH)
    instances = [FakeTaskInstance("task-0")]

    results = run_matched_replay(instances, grid, _stub_runner, dim=8, include_selection=False)

    assert len(results) == len(grid.factorial_cells)
    assert all(not r.condition_name.startswith("selection=") for r in results)
