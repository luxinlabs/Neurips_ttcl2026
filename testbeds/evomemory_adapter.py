"""TestbedAdapter implementation for Evo-Memory (arXiv 2511.20857, Google
DeepMind + UIUC). Repo link was found but not confirmed canonical in earlier
research (a third-party mirror at github.com/zhaosnw/evo_mem was located) —
verify the official link from the paper before depending on it.

STATUS: stub, same conformance contract as agentodyssey_adapter.py.

Note carried from PAPER_PLAN.md [INSIGHT: base_model_and_stack]: Evo-Memory's
own paper evaluated only closed-weight API models (Gemini-2.5, Claude
3.5/3.7) — there is no open-weight precedent to match here, so once this
adapter is wired to Llama-3.1-8B-Instruct, our numbers on this testbed will
not be directly comparable to Evo-Memory's own published baselines. That is
an acknowledged limitation, not a blocker: this testbed is used as a second
matched-replay stage for our own infra-noise ablation, not as a leaderboard.

TODOs pending real integration:
  1. Confirm and vendor the canonical Evo-Memory repo/dataset
  2. Map their 10 multi-turn goal-oriented / single-turn QA datasets to the
     matched-replay instance_id contract (same starting instance across
     all grid conditions)
  3. Decide how recall_probe() is derived from their existing QA structure
     (likely more direct than AgentOdyssey's, since Evo-Memory already has
     QA-style probes built into its 10 datasets)
"""
from __future__ import annotations

from driftbench.agent.interfaces import StepResult


class EvoMemoryAdapter:
    def __init__(self):
        raise NotImplementedError(
            "Evo-Memory integration pending: confirm the canonical repo link "
            "and wire its dataset-loading API here."
        )

    def reset(self, instance_id: str) -> str:
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def recall_probe(self) -> tuple[str, str]:
        raise NotImplementedError
