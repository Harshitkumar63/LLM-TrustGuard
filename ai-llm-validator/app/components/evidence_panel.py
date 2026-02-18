"""
evidence_panel.py — Streamlit component for rendering retrieved evidence.

Displays per-claim evidence passages, NLI verdicts, and factual scores
in an expandable accordion layout.

This file contains NO core logic.
"""

from typing import Any, Dict, List

import streamlit as st


def render_evidence_panel(
    claims: List[Dict[str, Any]],
    claim_results: List[Dict[str, Any]],
) -> None:
    """
    Render the evidence / claim-verification panel.

    Parameters
    ----------
    claims : list of dict
        Raw claims from the claim extractor.
    claim_results : list of dict
        NLI verification results — one per claim.
    """
    st.subheader("Claim-Level Evidence & Verification")

    if not claim_results:
        st.info("No claims were extracted from the LLM output.")
        return

    for idx, result in enumerate(claim_results, start=1):
        claim_text: str = result["claim"]
        factual_score: float = result["factual_score"]
        best_label: str = result["best_label"]

        # Colour-code by label
        label_colours = {
            "entailment": "🟢",
            "neutral": "🟡",
            "contradiction": "🔴",
        }
        icon = label_colours.get(best_label, "⚪")

        with st.expander(
            f"{icon} Claim {idx}: {claim_text[:90]}{'…' if len(claim_text) > 90 else ''}  "
            f"— Score: {factual_score:.2f}"
        ):
            st.markdown(f"**Full Claim:** {claim_text}")
            st.markdown(
                f"**NLI Verdict:** `{best_label}`  |  "
                f"**Factual Score:** `{factual_score:.3f}`"
            )

            if result.get("best_evidence"):
                st.markdown(f"**Best Evidence:** {result['best_evidence']}")

            # Per-evidence detail table
            per_ev = result.get("per_evidence", [])
            if per_ev:
                st.markdown("---")
                st.markdown("**All Retrieved Evidence:**")
                for eidx, ev in enumerate(per_ev, start=1):
                    probs = ev.get("probabilities", {})
                    st.markdown(
                        f"**{eidx}.** {ev['evidence_text'][:200]}  \n"
                        f"   Label: `{ev['label']}`  |  "
                        f"Entail: `{probs.get('entailment', 0):.3f}`  |  "
                        f"Neutral: `{probs.get('neutral', 0):.3f}`  |  "
                        f"Contradict: `{probs.get('contradiction', 0):.3f}`"
                    )
