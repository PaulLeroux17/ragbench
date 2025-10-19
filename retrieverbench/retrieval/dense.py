import logging
from typing import Any, Dict, Optional
import torch

from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval


def build_dense_model(
    model_name: str,
    max_length: int,
    query_prompt: Optional[str],
    passage_prompt: Optional[str],
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    kwargs: Dict[str, Any] = {
        "max_length": int(max_length),
        "trust_remote_code": True,
        "device": device,
    }
    if query_prompt is not None or passage_prompt is not None:
        kwargs["prompt_names"] = {
            "query": query_prompt,
            "passage": passage_prompt,
        }

    return models.SentenceBERT(model_name, **kwargs)


def dense_init(
    model,
    max_length,
    query_prompt,
    passage_prompt,
    batch_size,
    corpus_chunk_size,
    top_k,
):
    dense_model = build_dense_model(
        model, max_length, query_prompt, passage_prompt
    )
    searcher = DRES(
        dense_model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size
    )
    retriever = EvaluateRetrieval(
        searcher, score_function="cos_sim"
    )  # TODO: peut etre mettre ça plus haut
    retriever.top_k = top_k
    return retriever
