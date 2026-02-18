"""
config.py — Centralized configuration for the AI LLM Validator system.

All configurable constants (model names, thresholds, weights, paths) live
here so that every other module imports from a single source of truth.
No magic numbers should appear outside this file.
"""

from pathlib import Path
from typing import Dict, List

# ──────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "evaluation" / "results"

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
# Sentence-BERT model used for passage embedding & retrieval
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# RoBERTa MNLI model used for natural-language inference
NLI_MODEL_NAME: str = "roberta-large-mnli"

# Sentiment model used inside the bias analyzer
SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"

# spaCy language model for claim extraction
SPACY_MODEL_NAME: str = "en_core_web_sm"

# ──────────────────────────────────────────────
# Retriever Configuration
# ──────────────────────────────────────────────
FAISS_TOP_K: int = 3  # number of passages to retrieve per claim

# ──────────────────────────────────────────────
# NLI Label Mapping
# ──────────────────────────────────────────────
# The RoBERTa-MNLI label order is: contradiction (0), neutral (1), entailment (2)
NLI_LABELS: List[str] = ["contradiction", "neutral", "entailment"]

# Weights assigned to each NLI label when computing a factual score in [0, 1].
# entailment → high, neutral → mid, contradiction → low
NLI_LABEL_SCORES: Dict[str, float] = {
    "entailment": 1.0,
    "neutral": 0.5,
    "contradiction": 0.0,
}

# ──────────────────────────────────────────────
# Bias Analyzer Configuration
# ──────────────────────────────────────────────
# Protected-attribute keywords used for keyword-based bias detection
PROTECTED_ATTRIBUTE_KEYWORDS: List[str] = [
    "gender", "race", "religion", "ethnicity", "sexual orientation",
    "disability", "age", "nationality", "immigrant", "transgender",
    "muslim", "christian", "jewish", "hindu", "black", "white",
    "asian", "latino", "latina", "hispanic", "gay", "lesbian",
    "bisexual", "women", "men", "female", "male",
]

# Simple toxicity / hate-speech keywords (expanded list)
TOXICITY_KEYWORDS: List[str] = [
    "stupid", "idiot", "dumb", "hate", "kill", "ugly",
    "loser", "worthless", "pathetic", "disgusting", "terrible",
    "horrible", "awful", "inferior", "superior",
]

# Thresholds for risk level classification
BIAS_RISK_THRESHOLDS: Dict[str, float] = {
    "low": 0.3,
    "medium": 0.6,
    # anything above medium threshold is "High"
}

# ──────────────────────────────────────────────
# Trust Score Weights
# ──────────────────────────────────────────────
TRUST_WEIGHT_FACTUAL: float = 0.6
TRUST_WEIGHT_EVIDENCE: float = 0.3
TRUST_PENALTY_BIAS: float = 0.2

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_FILE: Path = PROJECT_ROOT / "validator.log"
LOG_LEVEL: str = "INFO"
