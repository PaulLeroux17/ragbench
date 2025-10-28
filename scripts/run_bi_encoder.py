from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Any

import logging
import numpy as np
import pandas as pd
import yaml

from beir import LoggingHandler
from retrieverbench.data import load_beir
from retrieverbench.eval import evaluate_metrics
from retrieverbench.retrieval.bi_encoder import bi_encoder_init


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bi-encoder retrieval (config-driven) with retrieverbench."
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/CONFIG.yaml)",
    )
    return ap.parse_args()


# ===================================================================
# Utils
# ===================================================================


def slugify(s: str) -> str:
    return (
        s.replace("/", "-")
        .replace(":", "-")
        .replace(" ", "_")
        .replace("@", "_")
        .replace("#", "_")
    )


def _choose_sort_metric(columns):
    for m in ["NDCG@10", "MAP@10", "MRR@10", "Recall@100"]:
        if m in columns:
            return m, False
    if "avg_time_per_query_ms" in columns:
        return "avg_time_per_query_ms", True
    return None, True


def _cfg_get(cfg: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    for k in keys:
        lk, uk = k.lower(), k.upper()
        if lk in cfg:
            return cfg[lk]
        if uk in cfg:
            return cfg[uk]
    return default


def _write_trec_runfile(results: Dict[str, Dict[str, float]], path: Path, run_name: str = "bi-encoder"):
    lines = []
    for qid, doc_scores in results.items():
        ranked = sorted(doc_scores.items(), key=lambda x: -x[1])
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            lines.append(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}
")
    path.write_text("".join(lines), encoding="utf-8")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        raise SystemExit(1)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Data
    DATASET: str = _cfg_get(cfg, "dataset")
    SPLIT: str = _cfg_get(cfg, "split", default="test")

    # Encoder backend selection
    PROVIDER: str = str(_cfg_get(cfg, "provider", default="beir"))  # "beir" or "hf_auto"

    # Model + prompts
    MODEL: str = _cfg_get(cfg, "model")
    QUERY_PROMPT: Optional[str] = _cfg_get(cfg, "query_prompt", "prompt_query", default=None)
    PASSAGE_PROMPT: Optional[str] = _cfg_get(cfg, "passage_prompt", "prompt_passage", default=None)
    MAX_LENGTH: int = int(_cfg_get(cfg, "max_length", "max_seq_length", default=512))

    # Retrieval
    TOP_K: int = int(_cfg_get(cfg, "top_k", default=1000))
    SCORE_FUNCTION: str = str(_cfg_get(cfg, "score_function", default="cos_sim"))  # or "dot"
    BATCH_SIZE: int = int(_cfg_get(cfg, "batch_size", default=128))
    CORPUS_CHUNK_SIZE: int = int(_cfg_get(cfg, "corpus_chunk_size", "CORPUS_CHUNK_SIZE", default=50_000))

    # HF-only extras
    POOLING: str = str(_cfg_get(cfg, "pooling", default="mean"))
    NORMALIZE: bool = bool(_cfg_get(cfg, "normalize_embeddings", default=True))
    DTYPE: str = str(_cfg_get(cfg, "dtype", default="auto"))  # "auto" | "float32" | "float16" | "bfloat16"

    # Limits
    MAX_QUERIES: Optional[Any] = _cfg_get(cfg, "max_queries", "MAX_QUERIES", default=None)
    if MAX_QUERIES in (None, "NONE", "Null", "null"):
        MAX_QUERIES = None
    else:
        MAX_QUERIES = int(MAX_QUERIES)

    # Output
    OUT_DIR = Path(_cfg_get(cfg, "out_dir", "OUT_DIR", default="results"))
    SAVE_RUNFILE: bool = bool(_cfg_get(cfg, "save_runfile", "SAVE_RUNFILE", default=True))
    PRINT_TOPK: int = int(_cfg_get(cfg, "print_topk", "PRINT_TOPK", default=0))

    # ===================================================================
    # Dataset
    # ===================================================================
    corpus, queries, qrels = load_beir(DATASET, split=SPLIT.lower())
    logging.info(
        f"Loaded BEIR '{DATASET}' split='{SPLIT}' |corpus|={len(corpus)} |queries|={len(queries)}"
    )

    if MAX_QUERIES is not None and MAX_QUERIES < len(queries):
        limited_qids = list(queries.keys())[:MAX_QUERIES]
        queries = {qid: queries[qid] for qid in limited_qids}
        logging.info(f"Limiting to MAX_QUERIES={MAX_QUERIES}")

    # -------------------------------------------------------------------
    # Retriever (bi-encoder in DRES)
    # -------------------------------------------------------------------
    retriever = bi_encoder_init(
        MODEL,
        MAX_LENGTH,
        QUERY_PROMPT,
        PASSAGE_PROMPT,
        BATCH_SIZE,
        CORPUS_CHUNK_SIZE,
        TOP_K,
        score_function=SCORE_FUNCTION,
        provider=PROVIDER,
        pooling=POOLING,
        normalize_embeddings=NORMALIZE,
        dtype=DTYPE,
    )

    # -------------------------------------------------------------------
    # Retrieval + timing
    # -------------------------------------------------------------------
    t0 = time.perf_counter()
    results = retriever.retrieve(corpus, queries)
    t1 = time.perf_counter()

    total_time_s = t1 - t0
    nq = max(1, len(queries))
    avg_ms = (total_time_s / nq) * 1000.0
    qps = nq / total_time_s if total_time_s > 0 else float("inf")
    logging.info(
        f"retrieve() total={total_time_s:.3f}s | avg={avg_ms:.2f}ms/query | QPS={qps:.2f}"
    )

    if PRINT_TOPK and PRINT_TOPK > 0:
        shown = 0
        for qid, doc_scores in results.items():
            print(f"[QID={qid}] {queries[qid]}")
            for rank, (did, score) in enumerate(
                sorted(doc_scores.items(), key=lambda x: -x[1])[:PRINT_TOPK],
                start=1,
            ):
                print(f"  {rank:>3}. {did}  score={score:.4f}")
            shown += 1
            if shown >= 3:
                break

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------
    metrics = evaluate_metrics(qrels, results)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = slugify(MODEL)
    base = f"{DATASET}__{model_slug}__ml{MAX_LENGTH}__exact__bs{BATCH_SIZE}__cc{CORPUS_CHUNK_SIZE}"

    if SAVE_RUNFILE:
        trec_path = OUT_DIR / f"{base}.run.trec"
        _write_trec_runfile(results, trec_path, run_name="bi-encoder")

    row = {
        "dataset": DATASET,
        "split": SPLIT,
        "backend": "exact",
        "provider": PROVIDER,
        "score_function": SCORE_FUNCTION,
        "top_k": TOP_K,
        "model": MODEL,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "corpus_chunk_size": CORPUS_CHUNK_SIZE,
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "total_time_s": round(total_time_s, 3),
        "avg_time_per_query_ms": round(avg_ms, 3),
        "qps": round(qps, 3),
        **metrics,
    }
    df = pd.DataFrame([row])

    sort_metric, ascending = _choose_sort_metric(df.columns)
    if sort_metric is not None:
        df = df.sort_values(sort_metric, ascending=ascending).reset_index(drop=True)

    csv_path = OUT_DIR / f"{DATASET}__{model_slug}__bi_encoder_single.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    ndcg_key = next((m for m in metrics.keys() if m.lower().startswith("ndcg@")), None)
    recall_key = next((m for m in metrics.keys() if m.lower().startswith("recall@")), None)
    parts = [f"[model={MODEL}, ml={MAX_LENGTH}, bs={BATCH_SIZE}]"]
    if ndcg_key:
        parts.append(f"{ndcg_key}={metrics[ndcg_key]:.4f}")
    if recall_key:
        parts.append(f"{recall_key}={metrics[recall_key]:.4f}")
    parts.append(f"avg_search_time={row['avg_time_per_query_ms']:.3f} ms")
    print(" | ".join(parts))

    print(f"
Saved CSV to: {csv_path}")
    if SAVE_RUNFILE:
        print(f"Saved TREC runfile to: {trec_path}")


if __name__ == "__main__":
    main()