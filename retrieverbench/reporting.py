from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def log_row(
    k1: float, b: float, ndcg10: float, recall100: float, avg_latency_ms: float
) -> None:
    print(
        f"[k1={k1}, b={b}] "
        f"NDCG@10={ndcg10:.4f} | "
        f"Recall@100={recall100:.4f} | "
        f"avg_latency={avg_latency_ms:.3f} ms"
    )


def choose_sort_metric(columns: Iterable[str]) -> Tuple[str | None, bool]:
    if "NDCG@10" in columns:
        return "NDCG@10", False
    if "Recall@100" in columns:
        return "Recall@100", False
    if "avg_latency_ms" in columns:
        return "avg_latency_ms", True
    return None, True


def pivot_grid(df: pd.DataFrame, metric: str, k1_axis, b_axis) -> pd.DataFrame:
    k1_axis_sorted = sorted(set(k1_axis))
    b_axis_sorted = sorted(set(b_axis))
    mat = (
        df.pivot_table(index="b", columns="k1", values=metric, aggfunc="max")
        .reindex(index=b_axis_sorted)
        .reindex(columns=k1_axis_sorted)
    )
    return mat


def plot_heatmap(
    df: pd.DataFrame, metric: str, k1_axis, b_axis, out_path: Path
) -> None:
    mat_df = pivot_grid(df, metric, k1_axis, b_axis)
    mat = mat_df.values
    k1_tick = list(mat_df.columns)
    b_tick = list(mat_df.index)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat, aspect="auto", origin="lower")
    plt.xticks(
        ticks=np.arange(len(k1_tick)),
        labels=[str(x) for x in k1_tick],
        rotation=45,
        ha="right",
    )
    plt.yticks(ticks=np.arange(len(b_tick)), labels=[str(x) for x in b_tick])
    plt.xlabel("k1")
    plt.ylabel("b")
    plt.title(f"Heatmap {metric}")
    plt.colorbar(im, label=metric)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)


def print_topk(df: pd.DataFrame, sort_metric: str | None, k: int = 5) -> None:
    """Affiche le top-k (k1, b) avec les 3 métriques."""
    header_metric = sort_metric if sort_metric is not None else "score"
    print(f"\n=== Top {k} by {header_metric} ===")

    display_order = ["NDCG@10", "Recall@100", "avg_latency_ms"]
    for rank, (_, row) in enumerate(df.head(k).iterrows(), start=1):
        parts = [f"{rank}. k1={row['k1']}, b={row['b']}"]
        for m in display_order:
            if m in row.index and pd.notna(row[m]):  # type: ignore
                if m == "avg_latency_ms":
                    parts.append(f"{m}={row[m]:.3f}")
                else:
                    parts.append(f"{m}={row[m]:.4f}")
        print(" | ".join(parts))
