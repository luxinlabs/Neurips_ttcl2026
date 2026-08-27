"""Real EmbeddingBackend: BAAI/bge-base-en-v1.5, per PAPER_PLAN.md
[INSIGHT: base_model_and_stack].

Requires `sentence-transformers` (not in requirements.txt yet — it pulls in
torch, which is a heavier dependency than the rest of the harness needs for
its currently-tested components; add it when wiring this in for real runs).

Not covered by the current test suite (tests use FakeEmbedder from fakes.py).
"""
from __future__ import annotations

import numpy as np


def _auto_device() -> str:
    """Prefer Apple GPU (MPS), then CUDA, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str | None = None):
        # Lazy import: sentence-transformers/torch only required if this
        # backend is actually instantiated.
        from sentence_transformers import SentenceTransformer

        self.device = device or _auto_device()
        self._model = SentenceTransformer(model_name, device=self.device)
        dim_fn = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        self.dim = int(dim_fn())
        self.model_name = model_name

    def embed_one(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)
