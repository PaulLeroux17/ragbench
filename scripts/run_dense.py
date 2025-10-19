from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import logging
import numpy as np
import pandas as pd
import yaml

from beir import LoggingHandler, util
from retrieverbench.data import load_beir
from retrieverbench.eval import evaluate_metrics
from retrieverbench.retrieval.dense import dense_init


# ===================================================================
# CLI
# ===================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dense retrieval (config-driven) with retrieverbench."
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
def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )


def slugify(s: str) -> str:
    return (
        s.replace("/", "-")
        .replace(":", "-")
        .replace(" ", "_")
        .replace("@", "_")
        .replace("#", "_")
    )


def _choose_sort_metric(columns):
    # préfère l’efficacité, sinon latence
    for m in ["NDCG@10", "MAP@10", "MRR@10", "Recall@100"]:
        if m in columns:
            return m, False  # desc
    if "avg_time_per_query_ms" in columns:
        return "avg_time_per_query_ms", True  # asc
    return None, True


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    args = parse_args()
    setup_logging()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        raise SystemExit(1)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    DATASET: str = cfg.get("dataset")
    SPLIT: str = cfg.get("split")

    MODEL: str = cfg["model"]
    QUERY_PROMPT: str = cfg.get("query_prompt")
    PASSAGE_PROMPT: str = cfg.get("passage_prompt")
    MAX_LENGTH: int = int(cfg.get("max_length"))

    TOP_K: int = int(cfg.get("top_k"))

    BATCH_SIZE: int = int(cfg.get("batch_size"))
    CORPUS_CHUNK_SIZE: int = int(cfg.get("CORPUS_CHUNK_SIZE", 50_000))
    MAX_QUERIES: Optional[int] = cfg.get("MAX_QUERIES", None)
    MAX_QUERIES = (
        None
        if MAX_QUERIES in (None, "NONE", "Null", "null")
        else int(MAX_QUERIES)
    )

    OUT_DIR = Path(cfg.get("OUT_DIR", "RESULTS"))
    SAVE_RUNFILE: bool = bool(cfg.get("SAVE_RUNFILE", True))
    PRINT_TOPK: int = int(cfg.get("PRINT_TOPK", 0))

    # ===================================================================
    # Dataset
    # ===================================================================
    corpus, queries, qrels = load_beir(DATASET, split=SPLIT.lower())
    logging.info(
        f"Loaded BEIR '{DATASET}' split='{SPLIT}' |corpus|={len(corpus)} |queries|={len(queries)}"
    )

    if MAX_QUERIES is not None and MAX_QUERIES < len(queries):
        # conserve l’ordre stable des clés si possible
        limited_qids = list(queries.keys())[:MAX_QUERIES]
        queries = {qid: queries[qid] for qid in limited_qids}
        logging.info(f"Limiting to MAX_QUERIES={MAX_QUERIES}")

    # -------------------------------------------------------------------
    # Retriever dense
    # -------------------------------------------------------------------
    retriever = dense_init(
        MODEL,
        MAX_LENGTH,
        QUERY_PROMPT,
        PASSAGE_PROMPT,
        BATCH_SIZE,
        CORPUS_CHUNK_SIZE,
        TOP_K,
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

    # Optionnel: afficher les top-k pour les quelques premières requêtes
    if PRINT_TOPK and PRINT_TOPK > 0:
        shown = 0
        for qid, doc_scores in results.items():
            print(f"\n[QID={qid}] {queries[qid]}")
            for rank, (did, score) in enumerate(
                sorted(doc_scores.items(), key=lambda x: -x[1])[:PRINT_TOPK],
                start=1,
            ):
                print(f"  {rank:>3}. {did}  score={score:.4f}")
            shown += 1
            if shown >= 3:  # ne pas spammer la console
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
        util.save_runfile(str(OUT_DIR / f"{base}.run.trec"), results)

    row = {
        "dataset": DATASET,
        "split": SPLIT,
        "backend": "exact",
        "score_function": "cos_sim",
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
        df = df.sort_values(sort_metric, ascending=ascending).reset_index(
            drop=True
        )

    csv_path = OUT_DIR / f"{DATASET}__{model_slug}__dense_single.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    ndcg_key = next(
        (m for m in metrics.keys() if m.lower().startswith("ndcg@")), None
    )
    recall_key = next(
        (m for m in metrics.keys() if m.lower().startswith("recall@")), None
    )
    parts = [f"[model={MODEL}, ml={MAX_LENGTH}, bs={BATCH_SIZE}]"]
    if ndcg_key:
        parts.append(f"{ndcg_key}={metrics[ndcg_key]:.4f}")
    if recall_key:
        parts.append(f"{recall_key}={metrics[recall_key]:.4f}")
    parts.append(f"avg_search_time={row['avg_time_per_query_ms']:.3f} ms")
    print(" | ".join(parts))

    print(f"\nSaved CSV to: {csv_path}")
    if SAVE_RUNFILE:
        print(f"Saved runfile to: {OUT_DIR / f'{base}.run.trec'}")


if __name__ == "__main__":
    main()
