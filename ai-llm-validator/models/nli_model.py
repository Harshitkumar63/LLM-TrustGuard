"""
nli_model.py — RoBERTa-MNLI Natural Language Inference model wrapper.

Provides a clean interface for running NLI prediction on (premise, hypothesis)
pairs.  The model is loaded lazily and cached.  Softmax probabilities are
returned alongside the predicted label so callers can derive confidence.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.config import NLI_LABELS, NLI_MODEL_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


class NLIModel:
    """
    Wrapper around a HuggingFace NLI sequence-classification model.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    labels : list of str
        Ordered label names matching the model's output indices.
    """

    def __init__(self, model_name: str = NLI_MODEL_NAME) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier (default: roberta-large-mnli).
        """
        self.model_name: str = model_name
        self.labels: List[str] = NLI_LABELS
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Lazy loader ------------------------------------------------
    def _load(self) -> None:
        """Load tokenizer and model if not already loaded."""
        if self._model is None:
            logger.info("Loading NLI model: %s (device=%s)", self.model_name, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()
            logger.info("NLI model loaded successfully.")

    # ---- Public API -------------------------------------------------
    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Run NLI prediction for a single (premise, hypothesis) pair.

        Parameters
        ----------
        premise : str
            The evidence / source text.
        hypothesis : str
            The claim to verify.

        Returns
        -------
        dict
            Mapping from label name → softmax probability, e.g.
            ``{"contradiction": 0.05, "neutral": 0.15, "entailment": 0.80}``.
        """
        self._load()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # shape (1, 3)

        # Softmax over label dimension
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        result: Dict[str, float] = {
            label: float(prob) for label, prob in zip(self.labels, probs)
        }
        logger.debug("NLI probabilities: %s", result)
        return result

    def predict_batch(
        self, pairs: List[Tuple[str, str]], batch_size: int = 8
    ) -> List[Dict[str, float]]:
        """
        Run NLI prediction on a batch of (premise, hypothesis) pairs.

        Parameters
        ----------
        pairs : list of (str, str)
            Each element is ``(premise, hypothesis)``.
        batch_size : int
            Number of pairs per forward pass.

        Returns
        -------
        list of dict
            One probability dict per pair.
        """
        self._load()
        assert self._tokenizer is not None and self._model is not None

        results: List[Dict[str, float]] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            inputs = self._tokenizer(
                premises,
                hypotheses,
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
                    {label: float(p) for label, p in zip(self.labels, prob_row)}
                )

        return results
