"""
trust_scorer.py — Aggregate a final trust score from verification signals.

Takes the factual consistency score, evidence retrieval strength, and
bias penalty to produce a single **Trust Score** on a 0–100 scale, along
with a structured breakdown that the UI can display.

Formula
-------
    trust_score = (  W_FACTUAL  * factual_score
                   + W_EVIDENCE * evidence_score
                   - P_BIAS     * bias_score     ) × 100

The result is clamped to [0, 100].
"""

from typing import Any, Dict, List

from utils.config import (
    TRUST_PENALTY_BIAS,
    TRUST_WEIGHT_EVIDENCE,
    TRUST_WEIGHT_FACTUAL,
)
from utils.helpers import normalize_score
from utils.logger import get_logger

logger = get_logger(__name__)


class TrustScorer:
    """
    Computes and structures the final trust score.

    Class-level attributes mirror config weights so they can be overridden
    in tests without modifying global config.
    """

    def __init__(
        self,
        w_factual: float = TRUST_WEIGHT_FACTUAL,
        w_evidence: float = TRUST_WEIGHT_EVIDENCE,
        p_bias: float = TRUST_PENALTY_BIAS,
    ) -> None:
        """
        Parameters
        ----------
        w_factual : float
            Weight for the factual consistency score.
        w_evidence : float
            Weight for the evidence retrieval score.
        p_bias : float
            Penalty coefficient for the bias score.
        """
        self.w_factual = w_factual
        self.w_evidence = w_evidence
        self.p_bias = p_bias

    # ---- Helper: Evidence Score ------------------------------------
    @staticmethod
    def compute_evidence_score(
        claim_results: List[Dict[str, Any]],
    ) -> float:
        """
        Derive an overall evidence-retrieval strength score.

        Uses the mean of the best FAISS similarity scores across claims.
        Falls back to 0.0 if no evidence was retrieved.

        Parameters
        ----------
        claim_results : list of dict
            Each dict must have ``per_evidence`` (list of dicts with
            ``"evidence_text"``).  Optionally, a ``"score"`` key can
            be present from the retriever.

        Returns
        -------
        float
            Evidence strength in [0, 1].
        """
        if not claim_results:
            return 0.0

        # Collect best per-claim FAISS score (stored in retriever output)
        best_scores: List[float] = []
        for cr in claim_results:
            per_ev = cr.get("per_evidence", [])
            if per_ev:
                # factual_score from NLI is a proxy for evidence quality too
                best_scores.append(max(e.get("factual_score", 0.0) for e in per_ev))

        return float(sum(best_scores) / len(best_scores)) if best_scores else 0.0

    # ---- Public API -------------------------------------------------
    def score(
        self,
        claim_results: List[Dict[str, Any]],
        bias_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the final trust score.

        Parameters
        ----------
        claim_results : list of dict
            Output of ``NLIChecker.check_claims`` — one dict per claim.
        bias_result : dict
            Output of ``BiasAnalyzer.analyze``.

        Returns
        -------
        dict
            ``{
                "trust_score": float,         # 0–100
                "factual_score": float,       # 0–1
                "evidence_score": float,      # 0–1
                "bias_score": float,          # 0–1
                "breakdown": {
                    "factual_contribution": float,
                    "evidence_contribution": float,
                    "bias_penalty": float,
                },
                "rating": str                # Trusted / Uncertain / Untrustworthy
            }``
        """
        # --- Component scores ---
        if claim_results:
            factual_score: float = sum(
                cr["factual_score"] for cr in claim_results
            ) / len(claim_results)
        else:
            factual_score = 0.0

        evidence_score: float = self.compute_evidence_score(claim_results)
        bias_score: float = bias_result.get("bias_score", 0.0)

        # --- Weighted aggregation ---
        raw: float = (
            self.w_factual * factual_score
            + self.w_evidence * evidence_score
            - self.p_bias * bias_score
        )
        trust_score: float = normalize_score(raw * 100, low=0.0, high=100.0)
        trust_score = round(trust_score, 1)

        # --- Human-readable rating ---
        if trust_score >= 70:
            rating = "Trusted"
        elif trust_score >= 40:
            rating = "Uncertain"
        else:
            rating = "Untrustworthy"

        result: Dict[str, Any] = {
            "trust_score": trust_score,
            "factual_score": round(factual_score, 4),
            "evidence_score": round(evidence_score, 4),
            "bias_score": round(bias_score, 4),
            "breakdown": {
                "factual_contribution": round(self.w_factual * factual_score * 100, 2),
                "evidence_contribution": round(self.w_evidence * evidence_score * 100, 2),
                "bias_penalty": round(self.p_bias * bias_score * 100, 2),
            },
            "rating": rating,
        }
        logger.info(
            "Trust score: %.1f / 100  (%s)  [fact=%.2f  ev=%.2f  bias=%.2f]",
            trust_score,
            rating,
            factual_score,
            evidence_score,
            bias_score,
        )
        return result
