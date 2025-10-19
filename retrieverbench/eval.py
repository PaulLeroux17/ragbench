from __future__ import annotations
from typing import Dict, Mapping
from beir.retrieval.evaluation import EvaluateRetrieval


def evaluate_metrics(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """
    Compute standard BEIR metrics for retrieval quality.

    Args:
        qrels: relevance labels, dict[qid][docid] -> int relevance
        results: retrieved scores, dict[qid][docid] -> float score

    Returns:
        Dict with NDCG@10, NDCG@100, Recall@10, Recall@100.
    """
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 10, 100, 1000]
    )
    return {
        "NDCG@10": float(ndcg.get("NDCG@10", 0.0)),
        "Recall@100": float(recall.get("Recall@100", 0.0)),
    }
