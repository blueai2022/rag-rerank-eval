"""Precision@1 / recall@k scoring, mirroring the Go rag.Eval / rag.EvalRerank harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .index import Candidate


@dataclass
class GoldCase:
    quotation: str
    gold_code: str


@dataclass
class EvalReport:
    total: int
    precision_at_1: float
    recall_at_k: float
    k: int

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "precision_at_1": self.precision_at_1,
            "recall_at_k": self.recall_at_k,
            "k": self.k,
        }


def _codes_equal(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def evaluate(gold: list[GoldCase], k: int, retrieve: Callable[[str], list[Candidate]]) -> EvalReport:
    """Scores `retrieve(quotation) -> top-k candidates` against the gold set."""
    if not gold:
        return EvalReport(total=0, precision_at_1=0.0, recall_at_k=0.0, k=k)

    p1 = 0
    recall = 0
    for case in gold:
        candidates = retrieve(case.quotation)
        if candidates and _codes_equal(candidates[0].code, case.gold_code):
            p1 += 1
        if any(_codes_equal(c.code, case.gold_code) for c in candidates):
            recall += 1

    n = len(gold)
    return EvalReport(
        total=n,
        precision_at_1=p1 / n,
        recall_at_k=recall / n,
        k=k,
    )
