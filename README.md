# rag-rerank-eval

A small experiment showing that **it is always important to start with an 
eval. A general-purpose cross-encoder reranker to a bi-encoder retrieval 
pipeline is not automatically a plus** — and accuracy can get measurably 
worse on  narrow, technical short-text domains.

## Running it

```bash
pip install -r requirements.txt
python scripts/run_eval.py --pool 20 --k 5
```

Flags:

- `--candidates` — path to the candidate corpus JSON (default
  `data/candidates.json`).
- `--gold` — path to the gold set JSON (default `data/gold_cases.json`).
- `--pool` — how many candidates the bi-encoder retrieves before reranking.
- `--k` — final lineup size scored for `precision_at_1` / `recall_at_k`.
- `--bi-encoder-model` — SentenceTransformer model id (default
  `sentence-transformers/all-MiniLM-L6-v2`).
- `--cross-encoder-model` — CrossEncoder model id (default
  `cross-encoder/ms-marco-MiniLM-L-6-v2`).

## What this measures

Two pipelines are compared on the same gold set:

- **Baseline** — bi-encoder cosine similarity, top-k directly.
- **Reranked** — bi-encoder retrieves a wider pool, a cross-encoder reranks
  that pool, top-k is taken from the reranked order.

## Results

Measured with the bundled sample data (83 ICD-10-CM candidates, 35 gold
cases), `all-MiniLM-L6-v2` bi-encoder, `ms-marco-MiniLM-L-6-v2` cross-encoder:

```
$ python scripts/run_eval.py --pool 20 --k 5
{
  "baseline": {"total": 35, "precision_at_1": 0.800, "recall_at_k": 1.000, "k": 5},
  "reranked":  {"total": 35, "precision_at_1": 0.657, "recall_at_k": 0.943, "k": 5}
}

$ python scripts/run_eval.py --pool 10 --k 3
{
  "baseline": {"total": 35, "precision_at_1": 0.800, "recall_at_k": 1.000, "k": 3},
  "reranked":  {"total": 35, "precision_at_1": 0.657, "recall_at_k": 0.914, "k": 3}
}
```

This mirrors a real finding from a production ICD-10 code retrieval pipeline:
reranking a widened candidate pool with an off-the-shelf MS MARCO
cross-encoder reduced both `precision@1` and `recall@k` compared to plain
cosine-similarity retrieval, on every pool/k configuration tried.

## Data

- `data/candidates.json` — 83 ICD-10-CM codes and their official short
  titles.
- `data/gold_cases.json` — generic `(quotation, gold_code)` pairs written for
  this repo, with no connection to any real patient data.
