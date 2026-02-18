"""
bias_analyzer.py — Hybrid bias detection (ML + rule-based).

Combines three complementary signals to produce an overall bias score:

1. **Sentiment skew** — strong negative sentiment may indicate biased
   framing.  Measured via a pretrained DistilBERT sentiment model.
2. **Protected-attribute keyword density** — counts how many tokens from
   a curated list of protected attributes appear in the text.
3. **Toxicity keyword density** — simple keyword matching against a
   toxicity word list.

The three signals are weighted and merged into a single bias score in
[0, 1].  A risk level ("Low" / "Medium" / "High") and a human-readable
explanation are returned.

Design decisions
----------------
* Combining ML sentiment with rule-based keyword matching provides
  reasonable coverage while remaining transparent and debuggable.
* In production, each signal could be replaced by more sophisticated
  models (e.g., Perspective API for toxicity).
"""

from typing import Any, Dict, List, Optional

from models.bias_model import BiasModel
from utils.config import (
    BIAS_RISK_THRESHOLDS,
    PROTECTED_ATTRIBUTE_KEYWORDS,
    TOXICITY_KEYWORDS,
)
from utils.helpers import normalize_score
from utils.logger import get_logger

logger = get_logger(__name__)


class BiasAnalyzer:
    """
    Hybrid bias analyzer that fuses ML sentiment with keyword heuristics.

    Attributes
    ----------
    bias_model : BiasModel
        Sentiment-analysis model wrapper.
    """

    # Weights for the three sub-scores (must sum to 1.0)
    _W_SENTIMENT: float = 0.4
    _W_PROTECTED: float = 0.3
    _W_TOXICITY: float = 0.3

    def __init__(self, bias_model: Optional[BiasModel] = None) -> None:
        """
        Parameters
        ----------
        bias_model : BiasModel, optional
            Shared model instance.  Created internally if not provided.
        """
        self._bias_model: BiasModel = bias_model or BiasModel()

    # ---- Private helpers -------------------------------------------
    @staticmethod
    def _keyword_density(text: str, keywords: List[str]) -> float:
        """
        Fraction of *keywords* that appear at least once in *text*.

        Returns a value in [0, 1].
        """
        if not keywords:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw in text_lower)
        return hits / len(keywords)

    @staticmethod
    def _classify_risk(score: float) -> str:
        """Map a bias score to a human-readable risk level."""
        if score < BIAS_RISK_THRESHOLDS["low"]:
            return "Low"
        if score < BIAS_RISK_THRESHOLDS["medium"]:
            return "Medium"
        return "High"

    def _build_explanation(
        self,
        sentiment_score: float,
        protected_density: float,
        toxicity_density: float,
        risk_level: str,
    ) -> str:
        """Compose a brief natural-language explanation of the bias analysis."""
        parts: List[str] = []
        if sentiment_score > 0.5:
            parts.append(
                f"Sentiment analysis detected notable negative framing "
                f"(score {sentiment_score:.2f})."
            )
        if protected_density > 0.05:
            parts.append(
                f"Text references protected attributes "
                f"(keyword density {protected_density:.2%})."
            )
        if toxicity_density > 0.02:
            parts.append(
                f"Potential toxicity keywords found "
                f"(keyword density {toxicity_density:.2%})."
            )
        if not parts:
            parts.append("No significant bias signals detected.")
        parts.append(f"Overall risk level: {risk_level}.")
        return " ".join(parts)

    # ---- Public API -------------------------------------------------
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze *text* for bias and return a structured report.

        Parameters
        ----------
        text : str
            The LLM output text to analyze.

        Returns
        -------
        dict
            ``{
                "bias_score": float,      # 0–1
                "risk_level": str,        # Low / Medium / High
                "explanation": str,
                "details": {
                    "sentiment_score": float,
                    "protected_attribute_density": float,
                    "toxicity_density": float,
                }
            }``
        """
        # --- Signal 1: Sentiment skew ---
        sentiment_probs: Dict[str, float] = self._bias_model.predict_sentiment(text)
        # Higher negative probability → higher bias signal
        sentiment_score: float = sentiment_probs.get("NEGATIVE", 0.0)

        # --- Signal 2: Protected-attribute keyword density ---
        protected_density: float = self._keyword_density(
            text, PROTECTED_ATTRIBUTE_KEYWORDS
        )

        # --- Signal 3: Toxicity keyword density ---
        toxicity_density: float = self._keyword_density(text, TOXICITY_KEYWORDS)

        # --- Aggregate ---
        raw_bias: float = (
            self._W_SENTIMENT * sentiment_score
            + self._W_PROTECTED * protected_density
            + self._W_TOXICITY * toxicity_density
        )
        bias_score: float = normalize_score(raw_bias)
        risk_level: str = self._classify_risk(bias_score)

        explanation: str = self._build_explanation(
            sentiment_score, protected_density, toxicity_density, risk_level
        )

        result: Dict[str, Any] = {
            "bias_score": round(bias_score, 4),
            "risk_level": risk_level,
            "explanation": explanation,
            "details": {
                "sentiment_score": round(sentiment_score, 4),
                "protected_attribute_density": round(protected_density, 4),
                "toxicity_density": round(toxicity_density, 4),
            },
        }
        logger.info(
            "Bias analysis: score=%.3f  risk=%s", bias_score, risk_level
        )
        return result
