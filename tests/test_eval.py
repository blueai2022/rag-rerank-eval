"""Offline tests for the metrics logic (no model downloads required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_rerank_eval.eval import GoldCase, evaluate
from rag_rerank_eval.index import Candidate


def test_precision_at_1_all_correct():
    gold = [GoldCase(quotation="q1", gold_code="A"), GoldCase(quotation="q2", gold_code="B")]

    def retrieve(q):
        return [Candidate(code="A", description="", score=1.0)] if q == "q1" else [
            Candidate(code="B", description="", score=1.0)
        ]

    report = evaluate(gold, k=1, retrieve=retrieve)
    assert report.precision_at_1 == 1.0
    assert report.recall_at_k == 1.0


def test_recall_at_k_without_top1():
    gold = [GoldCase(quotation="q1", gold_code="B")]

    def retrieve(_q):
        return [
            Candidate(code="A", description="", score=0.9),
            Candidate(code="B", description="", score=0.8),
        ]

    report = evaluate(gold, k=2, retrieve=retrieve)
    assert report.precision_at_1 == 0.0
    assert report.recall_at_k == 1.0


def test_empty_gold_set():
    report = evaluate([], k=5, retrieve=lambda _q: [])
    assert report.total == 0
    assert report.precision_at_1 == 0.0
    assert report.recall_at_k == 0.0


def test_case_insensitive_code_match():
    gold = [GoldCase(quotation="q1", gold_code="e11.9")]

    def retrieve(_q):
        return [Candidate(code="E11.9", description="", score=1.0)]

    report = evaluate(gold, k=1, retrieve=retrieve)
    assert report.precision_at_1 == 1.0


if __name__ == "__main__":
    test_precision_at_1_all_correct()
    test_recall_at_k_without_top1()
    test_empty_gold_set()
    test_case_insensitive_code_match()
    print("all tests passed")
