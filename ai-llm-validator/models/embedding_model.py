"""
embedding_model.py — Sentence-BERT embedding wrapper.

Encapsulates model loading and encoding so that the rest of the codebase
never deals with SentenceTransformer internals directly.  The model is
loaded lazily on first call and cached for subsequent use.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper around a SentenceTransformer model for generating dense
    vector embeddings of text passages.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    model : SentenceTransformer or None
        Lazily-loaded model instance.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier for the sentence-transformer.
        """
        self.model_name: str = model_name
        self._model: Optional[SentenceTransformer] = None

    # ---- Lazy loader ------------------------------------------------
    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully.")
        return self._model

    # ---- Public API -------------------------------------------------
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of text strings into dense embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to embed.
        batch_size : int
            Encoding batch size.
        show_progress : bool
            Whether to display a progress bar.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(len(texts), embedding_dim)``.
        """
        model = self._load_model()
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.debug("Encoded %d texts → shape %s", len(texts), embeddings.shape)
        return embeddings

    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
