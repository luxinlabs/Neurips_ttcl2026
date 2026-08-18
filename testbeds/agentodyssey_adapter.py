"""TestbedAdapter implementation for AgentOdyssey (arXiv 2606.24893,
github.com/agentodyssey/agentodyssey — verified public repo per earlier
research, not yet installed/inspected in this session).

STATUS: stub. Structurally conforms to driftbench.agent.interfaces.TestbedAdapter
so it plugs into episode_loop.run_self_feedback_episode and
replay.matched_replay.run_matched_replay without further changes once real
integration lands. The actual environment calls (task loading by seed,
action submission, observation parsing, recall-probe extraction from the
procedurally generated entity/world-state graph) are TODOs pending:
  1. `pip install agentodyssey` (or vendor the repo) into the ttcl env
  2. Reading their task-instance / seed API to satisfy the matched-replay
     requirement (same starting instance across all grid conditions)
  3. Deciding how recall_probe() samples a fact from the world-state graph
     with supporting context removed, per SeqMem-Eval's protocol

Do not treat this file as functional — every method raises NotImplementedError.
"""
from __future__ import annotations

from driftbench.agent.interfaces import StepResult


class AgentOdysseyAdapter:
    def __init__(self):
        raise NotImplementedError(
            "AgentOdyssey integration pending: install github.com/agentodyssey/"
            "agentodyssey and wire its task-loading API here."
        )

    def reset(self, instance_id: str) -> str:
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def recall_probe(self) -> tuple[str, str]:
        raise NotImplementedError
