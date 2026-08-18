from pathlib import Path

from driftbench.config import build_grid

GRID_PATH = Path(__file__).parent.parent / "configs" / "grid_default.yaml"


def test_default_grid_has_nine_factorial_cells():
    grid = build_grid(GRID_PATH)
    assert len(grid.factorial_cells) == 9  # 3 stochastic x 3 systematic


def test_selection_conditions_are_not_crossed_into_factorial():
    grid = build_grid(GRID_PATH)
    # Selection conditions are separate, one-at-a-time entries — not multiplied
    # into the 9-cell grid, per [INSIGHT: ablation_design].
    assert len(grid.selection_conditions) == 4
    for cond in grid.selection_conditions:
        assert cond.name.startswith("selection=")


def test_baseline_cell_is_exact_and_fresh():
    grid = build_grid(GRID_PATH)
    baseline = grid.baseline
    assert baseline.stochastic.kind == "flat"
    assert baseline.systematic.rebuild_cadence == 1


def test_selection_conditions_pin_stochastic_and_systematic_to_baseline():
    grid = build_grid(GRID_PATH)
    baseline = grid.baseline
    for cond in grid.selection_conditions:
        assert cond.stochastic == baseline.stochastic
        assert cond.systematic == baseline.systematic
