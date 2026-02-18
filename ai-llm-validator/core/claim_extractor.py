"""
claim_extractor.py — Extract atomic factual claims from LLM output.

Uses spaCy for sentence segmentation and named-entity recognition (NER).
Each sentence is treated as an atomic claim; a simple heuristic confidence
score is assigned based on the number of named entities present (more
entities → higher confidence that the sentence is a verifiable factual
statement rather than filler or opinion).

Design decisions
----------------
* spaCy is preferred over regex-based splitting because it handles
  abbreviations, titles, and edge-cases more reliably.
* Confidence is a lightweight heuristic — in production this could be
  replaced by a learned classifier that distinguishes factual claims
  from opinions / questions / hedged statements.
"""

from typing import Any, Dict, List, Optional

import spacy

from utils.config import SPACY_MODEL_NAME
from utils.helpers import build_claim_dict, normalize_score
from utils.logger import get_logger

logger = get_logger(__name__)


class ClaimExtractor:
    """
    Breaks a block of LLM-generated text into structured atomic claims.

    Attributes
    ----------
    nlp : spacy.language.Language
        Loaded spaCy pipeline.
    """

    def __init__(self, spacy_model: str = SPACY_MODEL_NAME) -> None:
        """
        Parameters
        ----------
        spacy_model : str
            Name of the spaCy model to load (must be installed).
        """
        self._spacy_model_name: str = spacy_model
        self._nlp: Optional[spacy.language.Language] = None

    # ---- Lazy loader ------------------------------------------------
    def _load(self) -> spacy.language.Language:
        """Load the spaCy model if not already in memory."""
        if self._nlp is None:
            logger.info("Loading spaCy model: %s", self._spacy_model_name)
            self._nlp = spacy.load(self._spacy_model_name)
            logger.info("spaCy model loaded successfully.")
        return self._nlp

    # ---- Public API -------------------------------------------------
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract atomic factual claims from *text*.

        Pipeline:
        1. Run spaCy on the full text.
        2. Iterate over sentences.
        3. For each sentence, collect named entities.
        4. Compute a heuristic confidence score.
        5. Return list of claim dicts.

        Parameters
        ----------
        text : str
            Raw LLM output text.

        Returns
        -------
        list of dict
            Each dict has keys ``claim_text``, ``entities``, ``confidence``.
        """
        nlp = self._load()
        doc = nlp(text)

        claims: List[Dict[str, Any]] = []

        for sent in doc.sents:
            sentence_text: str = sent.text.strip()
            if not sentence_text:
                continue  # skip blank lines

            # Collect unique entity texts for this sentence
            entities: List[str] = list(
                {ent.text for ent in sent.ents}
            )

            # Heuristic confidence:
            #   base 0.5  +  0.1 per entity (capped at 1.0)
            # Sentences with more named entities are more likely to be
            # verifiable factual claims.
            confidence: float = normalize_score(
                0.5 + 0.1 * len(entities), low=0.0, high=1.0
            )

            claims.append(
                build_claim_dict(
                    claim_text=sentence_text,
                    entities=entities,
                    confidence=confidence,
                )
            )

        logger.info(
            "Extracted %d claims from text (%d chars).",
            len(claims),
            len(text),
        )
        return claims
