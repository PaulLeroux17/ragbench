from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, OrderedDict
from collections import OrderedDict as _OrderedDict
from beir import util
from beir.datasets.data_loader import GenericDataLoader


def load_beir(
    dataset: str,
    split: str = "test",
    cache_dir: Optional[str] = None,
) -> Tuple[
    Dict[str, Dict[str, Any]], OrderedDict[str, str], Dict[str, Dict[str, int]]
]:
    """
    Download (if needed) and load a BEIR dataset.

    Args:
        dataset: BEIR dataset ID (e.g., "scidocs", "scifact",
                 "trec-covid", ...)
        split:   One of "train" | "dev" | "test" (depending on dataset
                 availability)
        cache_dir: Local directory to cache the downloaded/unzipped data
                   (default: ./datasets)

    Returns:
        corpus: dict[doc_id] -> {"title": str, "text": str, ...}
        queries: OrderedDict[qid] -> query text (deterministic order by qid)
        qrels: dict[qid] -> dict[doc_id] -> relevance label (int)
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    cache_root = Path(cache_dir) if cache_dir else Path("datasets")
    cache_root.mkdir(parents=True, exist_ok=True)

    data_dir = util.download_and_unzip(url, str(cache_root))
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(
        split=split
    )

    queries_sorted: OrderedDict[str, str] = _OrderedDict(
        sorted(queries.items(), key=lambda kv: kv[0])
    )

    return corpus, queries_sorted, qrels
