"""
app.py — Streamlit dashboard for the AI LLM Validator.

Launch with:
    streamlit run app/app.py

The UI collects a user question and an LLM-generated answer, runs the
full validation pipeline, and displays:
    • Trust Score (large number + rating)
    • Factual consistency breakdown
    • Evidence panel (per-claim)
    • Bias analysis meter

All display logic is delegated to dedicated components under
``app/components/``.  No core ML logic lives in this file.
"""

import sys
from pathlib import Path

# ── Make project root importable ──────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components.bias_meter import render_bias_meter
from app.components.evidence_panel import render_evidence_panel
from app.components.score_display import render_trust_score
from core.pipeline import ValidationPipeline

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI LLM Validator",
    page_icon="🛡️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Sidebar — Project Info
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ AI LLM Validator")
    st.markdown(
        """
        **Hallucination & Bias Detection System**

        This tool audits LLM-generated responses for:
        - ✅ Factual consistency
        - ⚖️ Bias risk
        - 🏅 Overall trustworthiness

        ---
        *Pipeline:* Claim Extraction → Evidence Retrieval (FAISS + SBERT)
        → NLI Verification (RoBERTa-MNLI) → Bias Analysis → Trust Score
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit · HuggingFace · FAISS")

# ──────────────────────────────────────────────
# Pipeline Singleton (cached so models load once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models — this may take a minute on first run…")
def get_pipeline() -> ValidationPipeline:
    """Instantiate and cache the validation pipeline."""
    return ValidationPipeline()

# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.title("🔍 LLM Output Validator")
st.markdown(
    "Enter a user question and the LLM's response below, then click "
    "**Validate Output** to run the full verification pipeline."
)

# --- Input fields ---
user_question = st.text_input(
    "User Question",
    placeholder="e.g., Where is the Eiffel Tower?",
)

llm_output = st.text_area(
    "LLM Output",
    height=180,
    placeholder="Paste the LLM-generated answer here…",
)

# --- Validate button ---
validate_clicked = st.button("🚀 Validate Output", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Validation Flow
# ──────────────────────────────────────────────
if validate_clicked:
    if not user_question.strip() or not llm_output.strip():
        st.warning("Please fill in both the user question and the LLM output.")
    else:
        pipeline = get_pipeline()

        with st.spinner("Running validation pipeline…"):
            report = pipeline.validate(user_question, llm_output)

        st.success(
            f"Validation complete in **{report['inference_time_s']:.2f}s**."
        )

        # ---- Trust Score ----
        st.markdown("---")
        render_trust_score(report["trust_result"])

        # ---- Evidence Panel ----
        st.markdown("---")
        render_evidence_panel(report["claims"], report["claim_results"])

        # ---- Bias Meter ----
        st.markdown("---")
        render_bias_meter(report["bias_result"])

        # ---- Raw Report (expandable) ----
        with st.expander("📄 Full Validation Report (JSON)"):
            st.json(report)
