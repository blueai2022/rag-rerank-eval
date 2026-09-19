#!/usr/bin/env python3
"""Compare bi-encoder-only retrieval against bi-encoder + cross-encoder rerank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_rerank_eval.eval import GoldCase, evaluate
from rag_rerank_eval.index import BiEncoderIndex
from rag_rerank_eval.reranker import CrossEncoderReranker

DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="data/candidates.json")
    parser.add_argument("--gold", default="data/gold_cases.json")
    parser.add_argument("--pool", type=int, default=20, help="bi-encoder pool width reranked down to k")
    parser.add_argument("--k", type=int, default=5, help="final lineup width")
    parser.add_argument("--bi-encoder-model", default=DEFAULT_BI_ENCODER)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    candidates_data = json.loads(Path(args.candidates).read_text())
    gold_data = json.loads(Path(args.gold).read_text())
    gold = [GoldCase(quotation=c["quotation"], gold_code=c["gold_code"]) for c in gold_data]
    if not gold:
        raise SystemExit("gold set is empty")

    codes = [c["code"] for c in candidates_data]
    descriptions = [c["description"] for c in candidates_data]

    print(f"Loading bi-encoder {args.bi_encoder_model}...", file=sys.stderr)
    bi_encoder = SentenceTransformer(args.bi_encoder_model)
    index = BiEncoderIndex(bi_encoder, codes, descriptions)

    baseline = evaluate(gold, args.k, lambda q: index.search(q, args.k))

    print(f"Loading cross-encoder {args.cross_encoder_model}...", file=sys.stderr)
    cross_encoder = CrossEncoder(args.cross_encoder_model)
    reranker = CrossEncoderReranker(cross_encoder)

    def retrieve_reranked(query: str):
        pool = index.search(query, args.pool)
        return reranker.rerank(query, pool, args.k)

    reranked = evaluate(gold, args.k, retrieve_reranked)

    print(json.dumps({"baseline": baseline.as_dict(), "reranked": reranked.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
