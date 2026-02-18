"""
evaluate.py — Run the validation pipeline on test cases and report metrics.

Usage
-----
    python -m evaluation.evaluate

The script:
1. Loads test cases from ``data/test_cases.json``.
2. Runs each through the validation pipeline.
3. Computes Precision / Recall / F1 by comparing predicted trust scores
   against the ``expected_trust`` labels.
4. Saves a JSON results file under ``evaluation/results/``.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path so relative imports resolve
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.pipeline import ValidationPipeline
from evaluation.metrics import compute_metrics
from utils.config import DATA_DIR, RESULTS_DIR
from utils.helpers import load_json, save_json
from utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation(
    test_cases_path: Path = DATA_DIR / "test_cases.json",
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Execute the full evaluation workflow.

    Parameters
    ----------
    test_cases_path : Path
        Path to the test-cases JSON file.
    results_dir : Path
        Directory where results JSON will be saved.

    Returns
    -------
    dict
        ``{ "metrics": {...}, "per_case": [...] }``
    """
    logger.info("=" * 60)
    logger.info("Starting evaluation run")
    logger.info("Test cases: %s", test_cases_path)

    test_cases: List[Dict[str, Any]] = load_json(test_cases_path)
    logger.info("Loaded %d test cases.", len(test_cases))

    # -- Initialize pipeline (models loaded once) --
    pipeline = ValidationPipeline()

    predictions: List[float] = []
    labels: List[str] = []
    per_case: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for idx, tc in enumerate(test_cases, start=1):
        tc_id: str = tc.get("id", f"tc_{idx:03d}")
        user_question: str = tc["user_question"]
        llm_output: str = tc["llm_output"]
        expected: str = tc.get("expected_trust", "unknown")

        logger.info(
            "[%d/%d] Evaluating %s — expected=%s",
            idx,
            len(test_cases),
            tc_id,
            expected,
        )

        report: Dict[str, Any] = pipeline.validate(user_question, llm_output)
        trust_score: float = report["trust_result"]["trust_score"]

        predictions.append(trust_score)
        labels.append(expected)

        per_case.append(
            {
                "id": tc_id,
                "user_question": user_question,
                "llm_output": llm_output[:200],
                "expected_trust": expected,
                "predicted_trust_score": trust_score,
                "rating": report["trust_result"]["rating"],
                "bias_score": report["bias_result"]["bias_score"],
                "inference_time_s": report["inference_time_s"],
            }
        )

    total_elapsed = time.perf_counter() - total_start

    # -- Compute aggregate metrics --
    metrics: Dict[str, Any] = compute_metrics(predictions, labels)
    metrics["total_time_s"] = round(total_elapsed, 2)

    # -- Save results --
    output: Dict[str, Any] = {
        "metrics": metrics,
        "per_case": per_case,
    }
    results_path = results_dir / "evaluation_results.json"
    save_json(output, results_path)
    logger.info("Results saved to %s", results_path)

    # -- Print summary --
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples       : {metrics['n_samples']}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1']:.4f}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Threshold     : {metrics['threshold']}")
    print(f"  Total Time    : {total_elapsed:.2f}s")
    print("=" * 60 + "\n")

    return output


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()
