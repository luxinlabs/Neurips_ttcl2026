"""LLMBackend on Apple Silicon GPU via MLX.

This is the Mac analog of VLLMClient: same generate() surface, but weights
run on Metal rather than CUDA. Default model is a 4-bit Llama-3.1-8B-Instruct
— the paper's planned AWQ-INT4 fallback when GPU budget is tight, applied
here because Metal has no vLLM path.
"""
from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise trading agent. "
    "When asked to trade, reply with exactly one line of the form "
    "'offer <good> in trade for ore', using the good named in the observation "
    "or in Relevant memory. When asked to write a memory note, reply with one "
    "sentence. When asked a question, reply with the short answer only. "
    "Do not add extra commentary."
)


class MLXClient:
    def __init__(
        self,
        model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        max_tokens: int = 96,
        temperature: float = 0.7,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._mx = mx
        self._model, self._tokenizer = load(model)
        self._sampler = make_sampler(temp=temperature)
        self.device = str(mx.default_device())

    def set_seed(self, seed: int) -> None:
        self._mx.random.seed(int(seed) & 0xFFFFFFFF)

    def generate(self, prompt: str) -> str:
        from mlx_lm import generate

        tokenizer = self._tokenizer
        messages = []
        if self.system_prompt and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
        text = generate(
            self._model,
            tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
        return (text or "").strip()

    def info(self) -> dict[str, Any]:
        return {
            "backend": "mlx",
            "model": self.model_name,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": bool(self.system_prompt),
        }
