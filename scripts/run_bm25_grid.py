from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd
import numpy as np
import yaml

from retrieverbench.data import load_beir
from retrieverbench.eval import evaluate_metrics
from retrieverbench.retrieval.bm25 import BM25Searcher
from retrieverbench.reporting import (
    log_row,
    choose_sort_metric,
    plot_heatmap,
    print_topk,
)


# ===================================================================
# CLI
# ===================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="BM25 grid search with config-driven parameters."
    )
    ap.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config (e.g., configs/scidocs_bm25_grid.yaml)",
    )
    return ap.parse_args()


def main() -> None:

    # ===================================================================
    # Get config file with cli
    # ===================================================================
    args = parse_args()
    if not args.config:
        print("Please provide a config via --config <path-to-yaml>.")
        print(
            "Example: python run_bm25_grid.py --config configs/scidocs_bm25_grid.yaml"
        )
        raise SystemExit(2)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        raise SystemExit(1)

    # ===================================================================
    # Load config / parameters
    # ===================================================================
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset: str = cfg["beir"]["dataset"]
    split: str = cfg["beir"].get("split", "test")

    index: str = cfg["pyserini"]["prebuilt_index"]
    k: int = int(cfg["retrieval"].get("k", 1000))
    rm3: bool = bool(cfg["retrieval"].get("rm3", False))

    k1_vals = list(cfg["search"]["k1"])
    b_vals = list(cfg["search"]["b"])

    out_csv = cfg.get("output", f"results/{dataset}_bm25_grid.csv")

    plots_cfg = cfg.get("plots", {})
    make_plots: bool = bool(plots_cfg.get("enabled", True))
    fig_dir = Path(plots_cfg.get("out_dir", "figures"))

    # ===================================================================
    # Load dataset
    # ===================================================================
    print("[INFO] Loading dataset…")
    corpus, queries, qrels = load_beir(dataset, split=split)
    print(f"[OK] Loaded. n_queries={len(queries)}")

    # ===================================================================
    # Initialize BM25 Searcher
    # ===================================================================
    print(f"[INFO] Initializing BM25 (index='{index}', rm3={rm3})")
    searcher = BM25Searcher(index, use_rm3=False)

    # ===================================================================
    # Grid search over k1 and b
    # ===================================================================
    all_rows: list[Dict[str, float]] = []

    for k1, b in itertools.product(k1_vals, b_vals):
        searcher.set_bm25(k1=k1, b=b)

        results: Mapping[str, Mapping[str, float]] = {}
        search_times: list[float] = []

        for qid, qtext in queries.items():
            t_start = time.perf_counter()
            hits = searcher.search(qtext, k)
            t_end = time.perf_counter()
            search_times.append(t_end - t_start)

            results[qid] = {h.docid: float(h.score) for h in hits}

        # Compute evaluation metrics (outside timing)
        metrics = evaluate_metrics(qrels, results)

        # Average search latency for this (k1, b) in ms
        avg_latency_ms = np.mean(search_times)

        # Collect row and log
        row = {
            "k1": float(k1),
            "b": float(b),
            "NDCG@10": float(metrics.get("NDCG@10", 0.0)),
            "Recall@100": float(metrics.get("Recall@100", 0.0)),
            "avg_latency_ms": float(avg_latency_ms),
        }
        all_rows.append(row)

        log_row(
            k1=float(k1),
            b=float(b),
            ndcg10=row["NDCG@10"],
            recall100=row["Recall@100"],
            avg_latency_ms=row["avg_latency_ms"],
        )

    # ===================================================================
    # Save results
    # ===================================================================
    df = pd.DataFrame(all_rows)

    sort_metric, ascending = choose_sort_metric(df.columns)
    if sort_metric is not None and not df.empty:
        df = df.sort_values(sort_metric, ascending=ascending).reset_index(
            drop=True
        )

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    # ===================================================================
    # Generate plots (only our 3 metrics)
    # ===================================================================
    if make_plots and len(df) > 0:
        for m in [
            x
            for x in ("NDCG@10", "Recall@100", "avg_latency_ms")
            if x in df.columns
        ]:
            plot_heatmap(
                df, m, k1_vals, b_vals, fig_dir / f"heatmap_{dataset}_{m}.png"
            )

    # ===================================================================
    # Console output: Top-5
    # ===================================================================
    if len(df) > 0:
        print_topk(df, sort_metric, k=5)

    print(f"\nSaved CSV to: {out_csv_path}")
    if make_plots:
        print(f"Figures saved to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
