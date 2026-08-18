"""Real LLMBackend: Llama-3.1-8B-Instruct served via vLLM's OpenAI-compatible
API, per PAPER_PLAN.md [INSIGHT: base_model_and_stack].

Requires a running vLLM server, e.g.:
    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

Not covered by the current test suite (tests use FakeLLM from fakes.py) —
this class needs a live server or a mocked HTTP layer to test meaningfully,
which is out of scope until the Lambda GPU allocation is provisioned.
"""
from __future__ import annotations


class VLLMClient:
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        # Lazy import: openai is only required if this backend is actually
        # instantiated, so the rest of the harness has no hard dependency on it.
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
