"""
score_display.py — Streamlit component for rendering the Trust Score.

Renders the large trust-score number, its rating badge, and the
contribution breakdown (factual / evidence / bias) as a set of
progress bars.

This file contains NO core logic — it only formats and displays data
provided by the pipeline.
"""

from typing import Any, Dict

import streamlit as st


def render_trust_score(trust_result: Dict[str, Any]) -> None:
    """
    Display the trust score section in the Streamlit UI.

    Parameters
    ----------
    trust_result : dict
        Trust-score dict returned by ``TrustScorer.score()``.
    """
    score: float = trust_result["trust_score"]
    rating: str = trust_result["rating"]

    # --- Colour based on rating ---
    colour_map = {
        "Trusted": "#28a745",        # green
        "Uncertain": "#ffc107",      # amber
        "Untrustworthy": "#dc3545",  # red
    }
    colour = colour_map.get(rating, "#6c757d")

    # --- Big score ---
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px;">
            <span style="font-size:72px; font-weight:bold; color:{colour};">
                {score:.0f}
            </span>
            <span style="font-size:24px; color:{colour};">/ 100</span>
            <br/>
            <span style="font-size:28px; color:{colour}; font-weight:600;">
                {rating}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Breakdown bars ---
    st.subheader("Score Breakdown")
    breakdown = trust_result["breakdown"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Factual Contribution", f"{breakdown['factual_contribution']:.1f}")
        st.progress(min(trust_result["factual_score"], 1.0))
    with col2:
        st.metric("Evidence Strength", f"{breakdown['evidence_contribution']:.1f}")
        st.progress(min(trust_result["evidence_score"], 1.0))
    with col3:
        st.metric("Bias Penalty", f"-{breakdown['bias_penalty']:.1f}")
        st.progress(min(trust_result["bias_score"], 1.0))
