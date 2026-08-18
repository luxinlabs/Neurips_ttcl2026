"""Deterministic fakes for LLMBackend / EmbeddingBackend / TestbedAdapter,
so episode_loop.py and its callers are testable without vLLM, bge-base-en-v1.5,
or the AgentOdyssey/Evo-Memory repos installed. Real backends (VLLMClient,
BGEEmbedder, concrete testbed adapters) implement the same interfaces —
swapping them in requires no change to episode_loop.py or matched_replay.py.
"""
from __future__ import annotations

import hashlib

import numpy as np

from driftbench.agent.interfaces import StepResult


class FakeLLM:
    """Returns a fixed response, or one keyed by a substring of the prompt,
    so tests can assert on specific branches without a real model."""

    def __init__(self, default_response: str = "look around", keyed_responses: dict[str, str] | None = None):
        self.default_response = default_response
        self.keyed_responses = keyed_responses or {}
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, response in self.keyed_responses.items():
            if key in prompt:
                return response
        return self.default_response


class FakeEmbedder:
    """Hash-based deterministic embedding: same text -> same vector, without
    a real embedding model. Not semantically meaningful, only used to prove
    wiring (retrieval, staleness, recall@k) works end-to-end."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode()).digest()
        raw = np.frombuffer(digest[: 4 * self.dim], dtype=np.uint8).astype(np.float32)
        raw = np.resize(raw, self.dim)
        norm = np.linalg.norm(raw)
        return raw / norm if norm > 0 else raw


class FakeTestbedAdapter:
    """A trivial N-step environment: always succeeds, ends after n_steps,
    with a fixed recall probe. Enough to exercise episode_loop's control flow."""

    def __init__(self, n_steps: int = 3, probe_question: str = "What trades ore for silver?", probe_answer: str = "the merchant"):
        self.n_steps = n_steps
        self.probe_question = probe_question
        self.probe_answer = probe_answer
        self._step_count = 0

    def reset(self, instance_id: str) -> str:
        self._step_count = 0
        return f"start of {instance_id}"

    def step(self, action: str) -> StepResult:
        self._step_count += 1
        done = self._step_count >= self.n_steps
        return StepResult(observation=f"obs after {action}", success=True, done=done)

    def recall_probe(self) -> tuple[str, str]:
        return self.probe_question, self.probe_answer
