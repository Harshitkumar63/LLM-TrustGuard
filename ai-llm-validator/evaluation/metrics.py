"""
metrics.py — Classification metrics for evaluating the validator.

Provides Precision, Recall, and F1 computation for the binary task of
determining whether an LLM output is **trustworthy** (positive) or
**untrustworthy** (negative).  A configurable threshold on the trust
score converts the continuous 0–100 signal into a binary prediction.
"""

from typing import Any, Dict, List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Threshold for binary classification
# ──────────────────────────────────────────────
DEFAULT_TRUST_THRESHOLD: float = 50.0  # trust_score >= 50 → "high" (trustworthy)


def _to_binary(
    predictions: List[float],
    labels: List[str],
    threshold: float = DEFAULT_TRUST_THRESHOLD,
) -> Tuple[List[int], List[int]]:
    """
    Convert continuous trust scores and string labels to binary vectors.

    Parameters
    ----------
    predictions : list of float
        Trust scores (0–100).
    labels : list of str
        Ground-truth labels — ``"high"`` (trustworthy) or ``"low"`` (untrustworthy).

    Returns
    -------
    (pred_binary, label_binary)
        Lists of 0/1 ints.  1 = trustworthy.
    """
    pred_binary = [1 if p >= threshold else 0 for p in predictions]
    label_binary = [1 if l.strip().lower() == "high" else 0 for l in labels]
    return pred_binary, label_binary


def compute_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP).  Returns 0.0 if denominator is zero."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN).  Returns 0.0 if denominator is zero."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_f1(precision: float, recall: float) -> float:
    """F1 = 2 * (P * R) / (P + R).  Returns 0.0 if denominator is zero."""
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def compute_metrics(
    predictions: List[float],
    labels: List[str],
    threshold: float = DEFAULT_TRUST_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute Precision, Recall, F1, and a confusion-matrix breakdown.

    Parameters
    ----------
    predictions : list of float
        Trust scores returned by the pipeline.
    labels : list of str
        Expected trust levels (``"high"`` or ``"low"``).
    threshold : float
        Trust score cutoff for the binary decision.

    Returns
    -------
    dict
        ``{
            "precision": float,
            "recall": float,
            "f1": float,
            "accuracy": float,
            "confusion_matrix": {"tp": int, "fp": int, "tn": int, "fn": int},
            "threshold": float,
            "n_samples": int,
        }``
    """
    pred_bin, label_bin = _to_binary(predictions, labels, threshold)

    tp = sum(p == 1 and l == 1 for p, l in zip(pred_bin, label_bin))
    fp = sum(p == 1 and l == 0 for p, l in zip(pred_bin, label_bin))
    tn = sum(p == 0 and l == 0 for p, l in zip(pred_bin, label_bin))
    fn = sum(p == 0 and l == 1 for p, l in zip(pred_bin, label_bin))

    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)
    f1 = compute_f1(precision, recall)
    accuracy = (tp + tn) / len(pred_bin) if pred_bin else 0.0

    metrics: Dict[str, Any] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "threshold": threshold,
        "n_samples": len(pred_bin),
    }

    logger.info(
        "Metrics — P=%.3f  R=%.3f  F1=%.3f  Acc=%.3f  (n=%d)",
        precision,
        recall,
        f1,
        accuracy,
        len(pred_bin),
    )
    return metrics
