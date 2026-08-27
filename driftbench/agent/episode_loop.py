"""Generic self-feedback episode loop: observe -> retrieve -> act -> reflect
-> write. This is the mechanism the whole paper is about — the reflect/write
step is where self-generated feedback enters memory, and the retrieve step is
where stochastic/systematic/selection noise (via NoisyMemoryStore) determines
what gets reinforced next.

Deliberately generic over LLMBackend / EmbeddingBackend / TestbedAdapter (see
interfaces.py) so it is fully testable with fakes now, independent of the
vLLM / bge-base-en-v1.5 / AgentOdyssey-Evo-Memory integration work still
pending per PAPER_PLAN.md [INSIGHT: base_model_and_stack].
"""
from __future__ import annotations

import numpy as np

from driftbench.agent.interfaces import EmbeddingBackend, LLMBackend, TestbedAdapter
from driftbench.eval.seqmem_protocol import Checkpoint, SeqMemTrace
from driftbench.index.noisy_memory import NoisyMemoryStore
from driftbench.metrics.drift import DriftTracker


def _reflect_prompt(observation: str, action: str, result_observation: str) -> str:
    return (
        f"Observation: {observation}\nAction taken: {action}\n"
        f"Result: {result_observation}\n"
        "In one sentence, write a memory note capturing what you learned:"
    )


def _act_prompt(observation: str, retrieved_notes: list[str]) -> str:
    notes = "\n".join(f"- {n}" for n in retrieved_notes) or "(none)"
    return f"Observation: {observation}\nRelevant memory:\n{notes}\nNext action:"


def run_self_feedback_episode(
    instance_id: str,
    adapter: TestbedAdapter,
    memory: NoisyMemoryStore,
    llm: LLMBackend,
    embedder: EmbeddingBackend,
    n_steps: int,
    checkpoint_every: int = 1,
    k_retrieve: int = 4,
) -> SeqMemTrace:
    trace = SeqMemTrace(condition_name="", task_instance_id=instance_id)
    drift = DriftTracker(reference_episode=0)

    observation = adapter.reset(instance_id)
    successes: list[bool] = []
    # Closed-book recall still needs the episode transcript (observations /
    # actions / results) but must not include retrieved memory notes — that's
    # SeqMem-Eval's "supporting context removed" condition. Stateless
    # generate() calls have no hidden conversation state, so we pass the
    # transcript explicitly.
    transcript: list[str] = [f"Observation: {observation}"]

    for step in range(n_steps):
        query_vec = embedder.embed_one(observation)
        report = memory.retrieve(query_vec, k=k_retrieve)

        action = llm.generate(_act_prompt(observation, report.retrieved_payloads))
        result = adapter.step(action)
        successes.append(result.success)
        transcript.append(f"Action: {action}")
        transcript.append(f"Result: {result.observation}")

        feedback = llm.generate(_reflect_prompt(observation, action, result.observation))
        feedback_vec = embedder.embed_one(feedback)
        memory.write(feedback_vec, payload=feedback)

        observation = result.observation

        if step % checkpoint_every == 0:
            proxy_success = sum(successes) / len(successes)
            probe_q, probe_gt = adapter.recall_probe()
            history = "\n".join(transcript)
            recall_response = llm.generate(
                f"Episode so far:\n{history}\n\nWithout retrieving from memory, answer: {probe_q}"
            )
            recall_score = 1.0 if probe_gt.strip().lower() in recall_response.strip().lower() else 0.0
            drift.record(step, np.array([proxy_success, 1 - proxy_success]))
            trace.add(
                Checkpoint(
                    episode=step,
                    proxy_success=proxy_success,
                    recall_score=recall_score,
                    drift_kl=0.0 if step == 0 else drift.drift_at(step),
                    staleness=report.staleness,
                    recall_at_k=report.recall_at_k,
                    retention_rate=report.retention_rate,
                )
            )

        if result.done:
            break

    return trace
