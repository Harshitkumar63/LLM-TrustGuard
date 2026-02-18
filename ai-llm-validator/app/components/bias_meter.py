"""
bias_meter.py — Streamlit component for rendering bias analysis results.

Shows the overall bias score as a coloured meter, the risk level badge,
a natural-language explanation, and the underlying sub-signal details.

This file contains NO core logic.
"""

from typing import Any, Dict

import streamlit as st


def render_bias_meter(bias_result: Dict[str, Any]) -> None:
    """
    Render the bias-analysis section of the dashboard.

    Parameters
    ----------
    bias_result : dict
        Output of ``BiasAnalyzer.analyze()``.
    """
    st.subheader("Bias Analysis")

    score: float = bias_result["bias_score"]
    risk: str = bias_result["risk_level"]
    explanation: str = bias_result["explanation"]

    # --- Colour-coded risk badge ---
    risk_colours = {
        "Low": "#28a745",
        "Medium": "#ffc107",
        "High": "#dc3545",
    }
    colour = risk_colours.get(risk, "#6c757d")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; padding:10px;
                        border:2px solid {colour}; border-radius:12px;">
                <span style="font-size:40px; font-weight:bold; color:{colour};">
                    {score:.2f}
                </span>
                <br/>
                <span style="font-size:18px; color:{colour};">
                    {risk} Risk
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(score, 1.0))

    with col2:
        st.markdown(f"**Explanation:** {explanation}")

        # Sub-signal details
        details = bias_result.get("details", {})
        if details:
            st.markdown("---")
            st.markdown("**Sub-signal breakdown:**")
            st.markdown(
                f"- Sentiment negativity: `{details.get('sentiment_score', 0):.4f}`"
            )
            st.markdown(
                f"- Protected-attribute density: "
                f"`{details.get('protected_attribute_density', 0):.4f}`"
            )
            st.markdown(
                f"- Toxicity keyword density: "
                f"`{details.get('toxicity_density', 0):.4f}`"
            )
