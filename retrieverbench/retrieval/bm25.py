from __future__ import annotations
from pyserini.search.lucene import LuceneSearcher


class BM25Searcher:
    """
    Thin wrapper around Pyserini's LuceneSearcher to:
    - initialize from a prebuilt BEIR-aligned index
    - set BM25 hyperparameters (k1, b) with or without an rm3
    - run searches and return raw scores
    """

    def __init__(self, prebuilt_index: str, use_rm3: bool = False) -> None:
        """
        Args:
            prebuilt_index: Pyserini prebuilt index name
                            (e.g., 'beir-v1.0.0-scidocs.flat')
            use_rm3: whether to enable RM3 pseudo-relevance feedback
        """
        self.searcher = LuceneSearcher.from_prebuilt_index(prebuilt_index)
        if use_rm3:
            self.searcher.set_rm3()

    def set_bm25(self, k1: float, b: float) -> None:
        """Set BM25 hyperparameters on the underlying searcher."""
        self.searcher.set_bm25(k1=k1, b=b)

    def search(self, query: str, k: int = 1000):
        """
        Execute a search and return Pyserini hits.

        Args:
            query: query text
            k: number of hits

        Returns:
            List of hits (each has .docid and .score)
        """
        return self.searcher.search(query, k)
