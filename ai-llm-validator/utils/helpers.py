"""
helpers.py — Shared helper functions used across the AI LLM Validator.

Keeps small, reusable utilities in one place so that core modules stay
focused on their primary responsibility.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# JSON I/O
# ──────────────────────────────────────────────
def load_json(path: Path) -> Any:
    """
    Load and parse a JSON file.

    Parameters
    ----------
    path : Path
        Absolute or project-relative path to the JSON file.

    Returns
    -------
    Any
        Parsed JSON content (dict, list, etc.).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.debug("Loaded JSON from %s (%d top-level items)", path, len(data) if isinstance(data, (list, dict)) else 1)
    return data


def save_json(data: Any, path: Path) -> None:
    """
    Serialize *data* to a JSON file, creating parent directories as needed.

    Parameters
    ----------
    data : Any
        JSON-serializable Python object.
    path : Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.debug("Saved JSON to %s", path)


# ──────────────────────────────────────────────
# Text Utilities
# ──────────────────────────────────────────────
def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate *text* to *max_length* characters, appending '…' if trimmed.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def normalize_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Clamp *value* to the [low, high] range.

    Parameters
    ----------
    value : float
        Raw score.
    low : float
        Minimum allowed value.
    high : float
        Maximum allowed value.

    Returns
    -------
    float
        Clamped value.
    """
    return max(low, min(high, value))


# ──────────────────────────────────────────────
# Timing Context Manager
# ──────────────────────────────────────────────
class Timer:
    """
    Simple wall-clock timer used for logging inference latency.

    Usage
    -----
    >>> with Timer("NLI inference") as t:
    ...     result = nli_model.predict(...)
    >>> print(t.elapsed)   # seconds as float
    """

    def __init__(self, label: str = "Operation"):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.info("%s completed in %.3f s", self.label, self.elapsed)


# ──────────────────────────────────────────────
# Structured Report Helpers
# ──────────────────────────────────────────────
def build_claim_dict(claim_text: str, entities: List[str], confidence: float) -> Dict[str, Any]:
    """
    Return a standardized claim dictionary.

    Parameters
    ----------
    claim_text : str
    entities : list of str
    confidence : float  (0.0 – 1.0)

    Returns
    -------
    dict
    """
    return {
        "claim_text": claim_text,
        "entities": entities,
        "confidence": normalize_score(confidence),
    }
