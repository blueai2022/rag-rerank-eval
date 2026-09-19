"""In-memory bi-encoder embedding index over a small text corpus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    code: str
    description: str
    score: float


class BiEncoderIndex:
    """Cosine-similarity search over sentence-transformer embeddings."""

    def __init__(self, model, codes: list[str], descriptions: list[str]) -> None:
        self._model = model
        self.codes = codes
        self.descriptions = descriptions
        vecs = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._unit_vecs = vecs / norms

    def search(self, query: str, k: int) -> list[Candidate]:
        qvec = self._model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0:
            return []
        scores = self._unit_vecs @ (qvec / qnorm)
        top_idx = np.argsort(-scores)[:k]
        return [
            Candidate(code=self.codes[i], description=self.descriptions[i], score=float(scores[i]))
            for i in top_idx
        ]
