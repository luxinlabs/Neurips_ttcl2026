"""Real EmbeddingBackend: BAAI/bge-base-en-v1.5, per PAPER_PLAN.md
[INSIGHT: base_model_and_stack].

Requires `sentence-transformers` (not in requirements.txt yet — it pulls in
torch, which is a heavier dependency than the rest of the harness needs for
its currently-tested components; add it when wiring this in for real runs).

Not covered by the current test suite (tests use FakeEmbedder from fakes.py).
"""
from __future__ import annotations

import numpy as np


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu"):
        # Lazy import: sentence-transformers/torch only required if this
        # backend is actually instantiated.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_one(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)
