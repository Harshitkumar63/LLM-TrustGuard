"""
retriever.py — FAISS-based evidence retrieval over a knowledge base.

Loads passages from a JSON knowledge base, encodes them with Sentence-BERT,
builds a FAISS flat-IP (inner-product / cosine) index, and provides a
``retrieve`` method that returns the top-k most similar passages for a
given query string.

Design decisions
----------------
* FAISS IndexFlatIP is used (after L2-normalisation) so that cosine
  similarity is the ranking metric — fast and exact for moderate-sized
  knowledge bases (< 100 k passages).
* Embeddings are cached inside the instance so that the knowledge base
  only needs to be encoded once.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from models.embedding_model import EmbeddingModel
from utils.config import DATA_DIR, FAISS_TOP_K
from utils.helpers import load_json
from utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceRetriever:
    """
    FAISS-powered passage retriever backed by Sentence-BERT embeddings.

    Attributes
    ----------
    knowledge_base : list of dict
        Loaded knowledge-base entries (each must have a ``"text"`` key).
    index : faiss.IndexFlatIP or None
        Built FAISS index.
    """

    def __init__(
        self,
        kb_path: Optional[Path] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: int = FAISS_TOP_K,
    ) -> None:
        """
        Parameters
        ----------
        kb_path : Path, optional
            Path to the knowledge-base JSON file.  Defaults to
            ``data/sample_knowledge_base.json``.
        embedding_model : EmbeddingModel, optional
            Shared embedding model instance (avoids duplicate loading).
        top_k : int
            Default number of passages to retrieve per query.
        """
        self.kb_path: Path = kb_path or (DATA_DIR / "sample_knowledge_base.json")
        self.top_k: int = top_k
        self._embedding_model: EmbeddingModel = embedding_model or EmbeddingModel()

        # State populated by ``build_index``
        self.knowledge_base: List[Dict[str, Any]] = []
        self._passages: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[faiss.IndexFlatIP] = None

    # ---- Index construction -----------------------------------------
    def build_index(self) -> None:
        """
        Load the knowledge base, encode all passages, and build a FAISS
        inner-product index with L2-normalised vectors (≡ cosine similarity).
        """
        logger.info("Building FAISS index from %s …", self.kb_path)

        # Step 1 – load KB
        self.knowledge_base = load_json(self.kb_path)
        self._passages = [entry["text"] for entry in self.knowledge_base]
        logger.info("Knowledge base contains %d passages.", len(self._passages))

        # Step 2 – encode
        self._embeddings = self._embedding_model.encode(self._passages)
        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(self._embeddings)

        # Step 3 – build index
        dim: int = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)
        logger.info(
            "FAISS index built: %d vectors, dim=%d.", self._index.ntotal, dim
        )

    # ---- Retrieval --------------------------------------------------
    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant passages for *query*.

        Parameters
        ----------
        query : str
            Natural-language query (typically a claim).
        top_k : int, optional
            Override the default top-k.

        Returns
        -------
        list of dict
            Each dict: ``{"text": str, "score": float, "metadata": dict}``.

        Raises
        ------
        RuntimeError
            If ``build_index`` has not been called first.
        """
        if self._index is None:
            raise RuntimeError(
                "FAISS index has not been built.  Call build_index() first."
            )

        k: int = top_k or self.top_k

        # Encode the query and normalise
        query_vec: np.ndarray = self._embedding_model.encode([query])
        faiss.normalize_L2(query_vec)

        # Search
        scores, indices = self._index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS padding when fewer results than k
            entry = self.knowledge_base[idx]
            results.append(
                {
                    "text": entry["text"],
                    "score": float(score),
                    "metadata": {
                        key: val
                        for key, val in entry.items()
                        if key != "text"
                    },
                }
            )

        logger.debug(
            "Retrieved %d passages for query (top score=%.3f).",
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results
