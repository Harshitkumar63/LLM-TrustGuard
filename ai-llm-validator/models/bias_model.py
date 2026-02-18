"""
bias_model.py — Sentiment-analysis model wrapper for the bias analyzer.

Uses a pretrained DistilBERT fine-tuned on SST-2 for sentiment classification.
The bias analyzer combines this signal with keyword heuristics to produce a
holistic bias score.
"""

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.config import SENTIMENT_MODEL_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


class BiasModel:
    """
    Thin wrapper around a HuggingFace sentiment-classification model
    used as an input signal for bias analysis.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    """

    def __init__(self, model_name: str = SENTIMENT_MODEL_NAME) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier for sentiment classification.
        """
        self.model_name: str = model_name
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Lazy loader ------------------------------------------------
    def _load(self) -> None:
        """Load tokenizer and model if not already in memory."""
        if self._model is None:
            logger.info("Loading bias/sentiment model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()
            logger.info("Bias/sentiment model loaded successfully.")

    # ---- Public API -------------------------------------------------
    def predict_sentiment(self, text: str) -> Dict[str, float]:
        """
        Return sentiment probabilities for *text*.

        Parameters
        ----------
        text : str
            Input text (typically the full LLM output or a segment).

        Returns
        -------
        dict
            ``{"NEGATIVE": float, "POSITIVE": float}`` with softmax probs.
        """
        self._load()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        labels = ["NEGATIVE", "POSITIVE"]
        result: Dict[str, float] = {
            label: float(p) for label, p in zip(labels, probs)
        }
        logger.debug("Sentiment probabilities: %s", result)
        return result

    def predict_sentiment_batch(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict[str, float]]:
        """
        Predict sentiment for a batch of texts.

        Parameters
        ----------
        texts : list of str
        batch_size : int

        Returns
        -------
        list of dict
        """
        self._load()
        assert self._tokenizer is not None and self._model is not None

        results: List[Dict[str, float]] = []
        labels = ["NEGATIVE", "POSITIVE"]

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for prob_row in probs:
                results.append(
                    {label: float(p) for label, p in zip(labels, prob_row)}
                )
        return results
