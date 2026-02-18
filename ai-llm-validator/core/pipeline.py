"""
pipeline.py — End-to-end validation pipeline orchestrator.

Wires together every core module into a single ``validate`` call:

    User Question + LLM Output
            ↓
    Claim Extraction  →  Evidence Retrieval  →  NLI Checking
            ↓                                        ↓
    Bias Analysis                              Factual Scores
            ↓                                        ↓
                     Trust Score Aggregation
                            ↓
                   Structured Validation Report

The orchestrator owns the module instances and manages their lifecycle.
It is the only object the UI layer imports from the core package.
"""

import time
from typing import Any, Dict, List, Optional

from core.bias_analyzer import BiasAnalyzer
from core.claim_extractor import ClaimExtractor
from core.nli_checker import NLIChecker
from core.retriever import EvidenceRetriever
from core.trust_scorer import TrustScorer
from models.bias_model import BiasModel
from models.embedding_model import EmbeddingModel
from models.nli_model import NLIModel
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationPipeline:
    """
    End-to-end validation pipeline for LLM outputs.

    Attributes
    ----------
    claim_extractor : ClaimExtractor
    retriever : EvidenceRetriever
    nli_checker : NLIChecker
    bias_analyzer : BiasAnalyzer
    trust_scorer : TrustScorer
    """

    def __init__(self) -> None:
        """
        Initialize all sub-modules.

        Model instances are shared where possible to avoid redundant loading.
        """
        # Shared model instances
        self._embedding_model = EmbeddingModel()
        self._nli_model = NLIModel()
        self._bias_model = BiasModel()

        # Core modules
        self.claim_extractor = ClaimExtractor()
        self.retriever = EvidenceRetriever(embedding_model=self._embedding_model)
        self.nli_checker = NLIChecker(nli_model=self._nli_model)
        self.bias_analyzer = BiasAnalyzer(bias_model=self._bias_model)
        self.trust_scorer = TrustScorer()

        self._index_built: bool = False

    # ---- Lazy index construction ------------------------------------
    def _ensure_index(self) -> None:
        """Build the FAISS index once on first validation request."""
        if not self._index_built:
            self.retriever.build_index()
            self._index_built = True

    # ---- Public API -------------------------------------------------
    def validate(
        self, user_question: str, llm_output: str
    ) -> Dict[str, Any]:
        """
        Run the full validation pipeline.

        Parameters
        ----------
        user_question : str
            The user's original question (used for context).
        llm_output : str
            The LLM-generated answer to validate.

        Returns
        -------
        dict
            A comprehensive validation report:
            ``{
                "user_question": str,
                "llm_output": str,
                "claims": [...],
                "claim_results": [...],
                "bias_result": {...},
                "trust_result": {...},
                "inference_time_s": float,
            }``
        """
        start_time = time.perf_counter()
        logger.info("=" * 60)
        logger.info("Starting validation pipeline")
        logger.info("Question: %s", user_question[:120])
        logger.info("LLM output length: %d chars", len(llm_output))

        # --- Step 1: Build FAISS index (once) ---
        self._ensure_index()

        # --- Step 2: Claim Extraction ---
        claims: List[Dict[str, Any]] = self.claim_extractor.extract(llm_output)
        logger.info("Claims extracted: %d", len(claims))

        # --- Step 3: Evidence Retrieval (per claim) ---
        evidence_map: Dict[str, List[Dict[str, Any]]] = {}
        for claim in claims:
            claim_text: str = claim["claim_text"]
            evidence_map[claim_text] = self.retriever.retrieve(claim_text)

        # --- Step 4: NLI Checking ---
        claim_results: List[Dict[str, Any]] = self.nli_checker.check_claims(
            claims, evidence_map
        )

        # --- Step 5: Bias Analysis ---
        bias_result: Dict[str, Any] = self.bias_analyzer.analyze(llm_output)

        # --- Step 6: Trust Score ---
        trust_result: Dict[str, Any] = self.trust_scorer.score(
            claim_results, bias_result
        )

        elapsed: float = time.perf_counter() - start_time
        logger.info(
            "Pipeline complete in %.2f s — Trust Score: %.1f / 100",
            elapsed,
            trust_result["trust_score"],
        )

        return {
            "user_question": user_question,
            "llm_output": llm_output,
            "claims": claims,
            "claim_results": claim_results,
            "bias_result": bias_result,
            "trust_result": trust_result,
            "inference_time_s": round(elapsed, 3),
        }
