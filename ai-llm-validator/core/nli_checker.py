"""
nli_checker.py — Natural Language Inference-based fact checker.

Given a claim and a list of retrieved evidence passages, this module uses a
RoBERTa-MNLI model to determine whether each passage **entails**,
**contradicts**, or is **neutral** w.r.t. the claim.  These per-passage
verdicts are aggregated into a single factual-consistency score in [0, 1].

Design decisions
----------------
* The highest entailment probability across all evidence passages is used as
  the representative factual score for a claim.  This mirrors the intuition
  that a single strong supporting passage is sufficient evidence.
* An optional threshold can be used to classify a claim as
  "supported" / "refuted" / "not enough info".
"""

from typing import Any, Dict, List, Optional

from models.nli_model import NLIModel
from utils.config import NLI_LABEL_SCORES
from utils.helpers import normalize_score
from utils.logger import get_logger

logger = get_logger(__name__)


class NLIChecker:
    """
    Verifies factual claims against retrieved evidence using NLI.

    Attributes
    ----------
    nli_model : NLIModel
        The underlying NLI model wrapper.
    """

    def __init__(self, nli_model: Optional[NLIModel] = None) -> None:
        """
        Parameters
        ----------
        nli_model : NLIModel, optional
            An existing model instance.  Created internally if not provided.
        """
        self._nli_model: NLIModel = nli_model or NLIModel()

    # ---- Public API -------------------------------------------------
    def check_claim(
        self, claim: str, evidence_passages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify a single *claim* against multiple evidence passages.

        For every (evidence, claim) pair the NLI model predicts a
        probability distribution over {contradiction, neutral, entailment}.
        The passage with the highest entailment probability is selected as
        the *best evidence*, and a factual score is derived from it.

        Parameters
        ----------
        claim : str
            The atomic factual claim to verify.
        evidence_passages : list of dict
            Each dict must have a ``"text"`` key (the passage text).

        Returns
        -------
        dict
            ``{
                "claim": str,
                "factual_score": float,   # 0–1
                "best_label": str,        # entailment / neutral / contradiction
                "best_evidence": str,
                "per_evidence": [...]     # detailed per-passage results
            }``
        """
        if not evidence_passages:
            logger.warning("No evidence provided for claim: %s", claim[:80])
            return {
                "claim": claim,
                "factual_score": 0.0,
                "best_label": "neutral",
                "best_evidence": "",
                "per_evidence": [],
            }

        per_evidence: List[Dict[str, Any]] = []
        best_factual: float = 0.0
        best_label: str = "neutral"
        best_evidence_text: str = ""

        for passage in evidence_passages:
            evidence_text: str = passage["text"]
            # NLI: premise = evidence, hypothesis = claim
            probs: Dict[str, float] = self._nli_model.predict(
                premise=evidence_text, hypothesis=claim
            )

            # Derive a scalar factual score from the probability distribution
            # score = sum( p(label) * weight(label) )
            factual_score: float = sum(
                probs[label] * NLI_LABEL_SCORES[label] for label in probs
            )
            factual_score = normalize_score(factual_score)

            # Determine the top label
            top_label: str = max(probs, key=probs.get)  # type: ignore[arg-type]

            per_evidence.append(
                {
                    "evidence_text": evidence_text,
                    "probabilities": probs,
                    "factual_score": factual_score,
                    "label": top_label,
                }
            )

            # Track the best (highest entailment) passage
            if factual_score > best_factual:
                best_factual = factual_score
                best_label = top_label
                best_evidence_text = evidence_text

        result: Dict[str, Any] = {
            "claim": claim,
            "factual_score": best_factual,
            "best_label": best_label,
            "best_evidence": best_evidence_text,
            "per_evidence": per_evidence,
        }
        logger.debug(
            "Claim verified: score=%.2f  label=%s  claim=%s",
            best_factual,
            best_label,
            claim[:60],
        )
        return result

    def check_claims(
        self,
        claims: List[Dict[str, Any]],
        evidence_map: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Batch-verify multiple claims.

        Parameters
        ----------
        claims : list of dict
            Each dict must have a ``"claim_text"`` key.
        evidence_map : dict
            Mapping ``claim_text → list of evidence dicts``.

        Returns
        -------
        list of dict
            One verification result per claim.
        """
        results: List[Dict[str, Any]] = []
        for claim_dict in claims:
            claim_text: str = claim_dict["claim_text"]
            passages = evidence_map.get(claim_text, [])
            result = self.check_claim(claim_text, passages)
            results.append(result)
        return results
