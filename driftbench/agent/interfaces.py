"""Interfaces separating the generic self-feedback loop (locked, testable now)
from the concrete base-model / embedding / testbed integrations (per
[INSIGHT: base_model_and_stack]: vLLM-served Llama-3.1-8B-Instruct,
BAAI/bge-base-en-v1.5). Mirrors the same injection pattern used in
replay/matched_replay.py — real backends drop in without changing the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class EmbeddingBackend(Protocol):
    dim: int

    def embed_one(self, text: str) -> np.ndarray: ...


@dataclass
class StepResult:
    observation: str
    success: bool
    done: bool


class TestbedAdapter(Protocol):
    """One episode's worth of environment interaction for a single matched
    task instance. Concrete implementations: testbeds/agentodyssey_adapter.py,
    testbeds/evomemory_adapter.py."""

    def reset(self, instance_id: str) -> str:
        """Load the matched starting state; return the initial observation."""
        ...

    def step(self, action: str) -> StepResult:
        ...

    def recall_probe(self) -> tuple[str, str]:
        """Return (probe_question, ground_truth_answer) for a free-form
        recall check with supporting context removed, per SeqMem-Eval's
        recall-vs-proxy protocol (arXiv 2605.15384)."""
        ...
