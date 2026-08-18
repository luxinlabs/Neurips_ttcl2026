"""Grid construction for the matched-replay ablation.

Implements [INSIGHT: ablation_design]: a 2-way factorial crossing stochastic x
systematic (3 levels each -> 9 cells), plus Selection kept one-at-a-time
against the shared (exact-search, fresh-index) baseline rather than crossed
into the grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from driftbench.index.approx_index import ApproxIndexConfig
from driftbench.index.eviction import EvictionConfig
from driftbench.index.shadow_index import ShadowIndexConfig


@dataclass(frozen=True)
class ConditionSpec:
    """One cell of the experimental design."""

    name: str
    stochastic: ApproxIndexConfig
    systematic: ShadowIndexConfig
    selection: EvictionConfig = field(default_factory=EvictionConfig)  # unbounded baseline


@dataclass(frozen=True)
class ExperimentGrid:
    factorial_cells: list[ConditionSpec]  # the 9-cell stochastic x systematic grid
    selection_conditions: list[ConditionSpec]  # one-at-a-time, secondary/appendix
    baseline_name: str  # e.g. "stochastic=exact,systematic=fresh"

    @property
    def baseline(self) -> ConditionSpec:
        """The (baseline, baseline) corner — doubles as the control condition
        and as the shared reference point for the Selection OFAT sweep."""
        return next(c for c in self.factorial_cells if c.name == self.baseline_name)


def _approx_levels(raw: dict) -> dict[str, ApproxIndexConfig]:
    levels = {}
    for name, params in raw.items():
        levels[name] = ApproxIndexConfig(**params)
    return levels


def _shadow_levels(raw: dict) -> dict[str, ShadowIndexConfig]:
    levels = {}
    for name, params in raw.items():
        levels[name] = ShadowIndexConfig(**params)
    return levels


def _eviction_conditions(raw: dict) -> dict[str, EvictionConfig]:
    conditions = {}
    for name, params in raw.items():
        conditions[name] = EvictionConfig(**params)
    return conditions


def build_grid(config_path: str | Path) -> ExperimentGrid:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    stochastic_levels = _approx_levels(raw["stochastic_levels"])
    systematic_levels = _shadow_levels(raw["systematic_levels"])
    selection_conditions = _eviction_conditions(raw["selection_conditions"])

    baseline_stochastic = raw["baseline"]["stochastic"]
    baseline_systematic = raw["baseline"]["systematic"]

    factorial_cells = [
        ConditionSpec(
            name=f"stochastic={s_name},systematic={t_name}",
            stochastic=stochastic_levels[s_name],
            systematic=systematic_levels[t_name],
        )
        for s_name in stochastic_levels
        for t_name in systematic_levels
    ]

    selection_cells = [
        ConditionSpec(
            name=f"selection={sel_name}",
            stochastic=stochastic_levels[baseline_stochastic],
            systematic=systematic_levels[baseline_systematic],
            selection=sel_config,
        )
        for sel_name, sel_config in selection_conditions.items()
    ]

    baseline_name = f"stochastic={baseline_stochastic},systematic={baseline_systematic}"
    return ExperimentGrid(
        factorial_cells=factorial_cells,
        selection_conditions=selection_cells,
        baseline_name=baseline_name,
    )
