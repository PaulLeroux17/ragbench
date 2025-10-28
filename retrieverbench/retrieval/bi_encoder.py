import logging
from typing import Any, Dict, Optional
import torch

from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval


# ---------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------

def build_beir_sentencebert(
    model_name: str,
    max_length: int,
    query_prompt: Optional[str] = None,
    passage_prompt: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """Build BEIR's SentenceBERT wrapper (uses beir.retrieval.models)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    kwargs: Dict[str, Any] = {
        "max_length": int(max_length),
        "trust_remote_code": True,
        "device": device,
    }
    if query_prompt is not None or passage_prompt is not None:
        kwargs["prompt_names"] = {"query": query_prompt, "passage": passage_prompt}

    return models.SentenceBERT(model_name, **kwargs)


def build_hf_auto_encoder(
    model_name: str,
    max_length: int,
    query_prompt: Optional[str],
    passage_prompt: Optional[str],
    pooling: str = "mean",
    normalize_embeddings: bool = True,
    dtype: str = "auto",
):
    """Build our lightweight HuggingFace -> BEIR adapter (no beir.models).

    Requires: retrieverbench.embeddings.hf_auto_encoder.HFAutoEncoder
    """
    from retrieverbench.embeddings.hf_auto_encoder import HFAutoEncoder

    return HFAutoEncoder(
        model_name=model_name,
        max_length=max_length,
        query_prompt=query_prompt,
        passage_prompt=passage_prompt,
        pooling=pooling,
        normalize=normalize_embeddings,
        dtype=dtype,
    )


# ---------------------------------------------------------------------
# Retriever factory (select provider)
# ---------------------------------------------------------------------

def bi_encoder_init(
    model_name: str,
    max_length: int,
    query_prompt: Optional[str],
    passage_prompt: Optional[str],
    batch_size: int,
    corpus_chunk_size: int,
    top_k: int,
    score_function: str = "cos_sim",
    provider: str = "beir",
    pooling: str = "mean",
    normalize_embeddings: bool = True,
    dtype: str = "auto",
) -> EvaluateRetrieval:
    """Create a BEIR `EvaluateRetrieval` using exact search (DRES).

    `provider` controls which embedding backend is used:
      - "beir": beir.retrieval.models.SentenceBERT
      - "hf_auto": raw Hugging Face (AutoModel) via our adapter
    """
    if provider.lower() == "hf_auto":
        encoder = build_hf_auto_encoder(
            model_name,
            max_length,
            query_prompt,
            passage_prompt,
            pooling=pooling,
            normalize_embeddings=normalize_embeddings,
            dtype=dtype,
        )
    else:
        encoder = build_beir_sentencebert(
            model_name, max_length, query_prompt, passage_prompt
        )

    searcher = DRES(
        encoder,
        batch_size=int(batch_size),
        corpus_chunk_size=int(corpus_chunk_size),
    )

    retriever = EvaluateRetrieval(
        searcher,
        score_function=score_function,
        k_values=[int(top_k)],
    )
    return retriever