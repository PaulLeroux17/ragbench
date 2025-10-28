from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Union, Optional

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel


@dataclass
class _HFConfig:
    model_name: str
    max_length: int = 512
    query_prompt: Optional[str] = None
    passage_prompt: Optional[str] = None
    pooling: str = "mean"  # "mean" | "cls"
    normalize: bool = True
    dtype: str = "auto"  # "auto" | "float32" | "float16" | "bfloat16"


class HFAutoEncoder:
    """
    Minimal Hugging Face encoder adapter implementing BEIR's expected
    interface: `encode_queries` and `encode_corpus` returning np.ndarray.

    Works with any Transformer encoder that produces `last_hidden_state`.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        query_prompt: Optional[str] = None,
        passage_prompt: Optional[str] = None,
        pooling: str = "mean",
        normalize: bool = True,
        dtype: str = "auto",
    ):
        self.cfg = _HFConfig(
            model_name=model_name,
            max_length=max_length,
            query_prompt=query_prompt,
            passage_prompt=passage_prompt,
            pooling=pooling,
            normalize=normalize,
            dtype=dtype,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        # "auto" -> let HF choose

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.cfg.model_name, trust_remote_code=True, torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

    # ------------------------- utils -------------------------

    def _apply_prompt(self, text: str, role: str) -> str:
        if role == "query" and self.cfg.query_prompt:
            return f"{self.cfg.query_prompt}: {text}" if self.cfg.query_prompt else text
        if role == "passage" and self.cfg.passage_prompt:
            return f"{self.cfg.passage_prompt}: {text}" if self.cfg.passage_prompt else text
        return text

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling.lower() == "cls":
            return last_hidden_state[:, 0]
        # mean pooling (mask-aware)
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / counts

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        out_embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**tokens)
            if not hasattr(outputs, "last_hidden_state"):
                raise RuntimeError("Model outputs have no last_hidden_state; unsupported architecture for pooling.")
            pooled = self._pool(outputs.last_hidden_state, tokens["attention_mask"])  # (B, H)
            if self.cfg.normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            out_embs.append(pooled.detach().cpu().numpy())
        return np.concatenate(out_embs, axis=0) if out_embs else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

    # ------------------------- BEIR API -------------------------

    def encode_queries(self, queries: List[str], batch_size: int = 128, **kwargs) -> np.ndarray:
        texts = [self._apply_prompt(q, role="query") for q in queries]
        return self._encode_texts(texts, batch_size=batch_size)

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, Dict[str, str]]],
        batch_size: int = 128,
        **kwargs,
    ) -> np.ndarray:
        # Accept list of dicts or dict id->dict
        if isinstance(corpus, dict):
            docs = corpus.values()
        else:
            docs = corpus
        texts: List[str] = []
        for d in docs:
            title = d.get("title", "") if isinstance(d, dict) else ""
            text = d.get("text", "") if isinstance(d, dict) else str(d)
            combined = (title + " 
" + text).strip() if title else text
            texts.append(self._apply_prompt(combined, role="passage"))
        return self._encode_texts(texts, batch_size=batch_size)
