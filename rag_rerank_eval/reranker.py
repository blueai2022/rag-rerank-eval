"""Cross-encoder reranking of a candidate pool."""

from __future__ import annotations

from .index import Candidate


class CrossEncoderReranker:
    """Scores (query, description) pairs and returns candidates sorted by relevance."""

    def __init__(self, model) -> None:
        self._model = model

    def rerank(self, query: str, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        pairs = [(query, c.description) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [
            Candidate(code=c.code, description=c.description, score=float(s))
            for c, s in ranked[:k]
        ]
