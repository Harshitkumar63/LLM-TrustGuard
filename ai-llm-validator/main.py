"""
main.py — CLI entry point for the AI LLM Validator.

Provides two sub-commands:

    python main.py validate   — Run a single validation from the terminal.
    python main.py evaluate   — Run the full evaluation suite.

If no arguments are given, a quick demo validation is executed.
"""

import json
import sys
from pathlib import Path

# ── Ensure project root is importable ─────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.pipeline import ValidationPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def demo_validation() -> None:
    """
    Run a quick demo validation with hard-coded inputs and print the
    structured report to stdout.
    """
    user_question = "Where is the Eiffel Tower located?"
    llm_output = (
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
        "It was constructed in 1889 and stands 330 metres tall. "
        "The tower was designed by Gustave Eiffel's engineering company."
    )

    print("=" * 60)
    print("AI LLM Validator — Demo Validation")
    print("=" * 60)
    print(f"Question : {user_question}")
    print(f"LLM Output: {llm_output[:120]}…")
    print("-" * 60)

    pipeline = ValidationPipeline()
    report = pipeline.validate(user_question, llm_output)

    # Pretty-print the trust result
    trust = report["trust_result"]
    print(f"\n  Trust Score : {trust['trust_score']:.1f} / 100  ({trust['rating']})")
    print(f"  Factual     : {trust['factual_score']:.3f}")
    print(f"  Evidence    : {trust['evidence_score']:.3f}")
    print(f"  Bias        : {trust['bias_score']:.3f}")
    print(f"  Time        : {report['inference_time_s']:.2f}s")
    print("-" * 60)

    # Per-claim summary
    for idx, cr in enumerate(report["claim_results"], start=1):
        print(
            f"  Claim {idx}: [{cr['best_label']:>13s}] "
            f"score={cr['factual_score']:.2f}  "
            f"{cr['claim'][:70]}"
        )

    # Bias summary
    bias = report["bias_result"]
    print(f"\n  Bias Risk : {bias['risk_level']}  (score={bias['bias_score']:.3f})")
    print(f"  {bias['explanation']}")
    print("=" * 60)


def run_evaluation_cli() -> None:
    """Delegate to the evaluation module."""
    from evaluation.evaluate import run_evaluation

    run_evaluation()


# ──────────────────────────────────────────────
# CLI Dispatcher
# ──────────────────────────────────────────────
def main() -> None:
    """Parse simple CLI arguments and dispatch."""
    args = sys.argv[1:]

    if not args or args[0] == "validate":
        demo_validation()
    elif args[0] == "evaluate":
        run_evaluation_cli()
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: python main.py [validate|evaluate]")
        sys.exit(1)


if __name__ == "__main__":
    main()
