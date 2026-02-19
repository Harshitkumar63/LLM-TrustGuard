# 🛡️ AI LLM Validator — Hallucination & Bias Detection System

A **production-grade, modular Python system** that evaluates Large Language Model outputs for **factual consistency** (hallucination detection), **bias risk**, and overall **trustworthiness** (0–100 trust score).

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)  
2. [Architecture](#architecture)  
3. [Project Structure](#project-structure)  
4. [Setup Instructions](#setup-instructions)  
5. [How to Run](#how-to-run)  
6. [Example Output](#example-output)  
7. [Evaluation](#evaluation)  
8. [Limitations](#limitations)  
9. [Future Scope](#future-scope)  

---

 Problem Statement

Large Language Models (GPT-4, Claude, Llama, etc.) can generate fluent, convincing text that contains **factual errors** (hallucinations) or **biased framing**.  When these outputs reach end-users without verification, trust erodes and real harm can follow—especially in healthcare, legal, and educational applications.

**AI LLM Validator** is a **post-generation verification layer** that sits between the LLM and the user.  It:

- Extracts atomic factual claims from the LLM's response.
- Retrieves evidence from a knowledge base using semantic search.
- Verifies each claim via Natural Language Inference.
- Detects bias through hybrid ML + rule-based analysis.
- Aggregates everything into a transparent **Trust Score (0–100)**.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
LLM Output (pasted / API)
      │
      ▼
┌─────────────────────────┐
│   Claim Extraction      │  ← spaCy NER + sentence segmentation
│   (claim_extractor.py)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Evidence Retrieval    │  ← FAISS + Sentence-BERT embeddings
│   (retriever.py)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   NLI Verification      │  ← RoBERTa-MNLI (entail / contradict / neutral)
│   (nli_checker.py)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Bias Analysis         │  ← Sentiment (DistilBERT) + Keyword heuristics
│   (bias_analyzer.py)    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Trust Score           │  ← Weighted aggregation (0–100)
│   (trust_scorer.py)     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Streamlit Dashboard   │  ← Interactive UI
│   (app/app.py)          │
└─────────────────────────┘
```

---

## 📁 Project Structure

```
ai-llm-validator/
│
├── app/                        # Streamlit UI layer
│   ├── app.py                  # Main dashboard entry point
│   └── components/
│       ├── score_display.py    # Trust score visualisation
│       ├── evidence_panel.py   # Per-claim evidence accordion
│       └── bias_meter.py       # Bias score + risk display
│
├── core/                       # Core business logic
│   ├── claim_extractor.py      # spaCy-based claim extraction
│   ├── retriever.py            # FAISS evidence retrieval
│   ├── nli_checker.py          # RoBERTa-MNLI NLI verification
│   ├── bias_analyzer.py        # Hybrid bias detection
│   ├── trust_scorer.py         # Weighted score aggregation
│   └── pipeline.py             # End-to-end orchestrator
│
├── models/                     # Model wrappers (lazy-load, cached)
│   ├── embedding_model.py      # Sentence-BERT wrapper
│   ├── nli_model.py            # RoBERTa-MNLI wrapper
│   └── bias_model.py           # DistilBERT sentiment wrapper
│
├── data/                       # Sample data & knowledge base
│   ├── sample_knowledge_base.json
│   ├── fever_samples.json
│   └── test_cases.json
│
├── evaluation/                 # Evaluation & metrics
│   ├── metrics.py              # Precision / Recall / F1
│   ├── evaluate.py             # Evaluation runner
│   └── results/                # Auto-generated results
│
├── utils/                      # Shared utilities
│   ├── config.py               # Centralised configuration
│   ├── logger.py               # Logging setup
│   └── helpers.py              # JSON I/O, timers, etc.
│
├── requirements.txt
├── README.md
└── main.py                     # CLI entry point
```

---

## ⚙️ Setup Instructions

### Prerequisites

- **Python 3.10+**
- **pip** (or conda)
- ~2 GB disk space for pretrained models (downloaded on first run)

### 1. Clone / Copy the Project

```bash
cd ai-llm-validator
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. (Optional) Verify Installation

```bash
python -c "import torch, transformers, sentence_transformers, faiss, spacy, streamlit; print('All imports OK')"
```

---

## 🚀 How to Run

### CLI — Quick Demo Validation

```bash
python main.py validate
```

Runs a hard-coded demo question + answer through the full pipeline and prints the Trust Score, per-claim verdicts, and bias analysis to the terminal.

### CLI — Full Evaluation Suite

```bash
python main.py evaluate
```

Loads all test cases from `data/test_cases.json`, runs each through the pipeline, computes Precision / Recall / F1, and saves results to `evaluation/results/evaluation_results.json`.

### Streamlit Dashboard

```bash
streamlit run app/app.py
```

Opens an interactive web UI where you can paste any question + LLM output and get a visual trust report.

---

## 📊 Example Output

### Trust Score

```
  Trust Score : 78.5 / 100  (Trusted)
  Factual     : 0.872
  Evidence    : 0.681
  Bias        : 0.043
  Time        : 3.21s
```

### Per-Claim Verification

```
  Claim 1: [    entailment] score=0.91  The Eiffel Tower is located in Paris, France.
  Claim 2: [    entailment] score=0.87  It was constructed in 1889.
  Claim 3: [       neutral] score=0.54  The tower was designed by Gustave Eiffel's company.
```

### Bias Analysis

```
  Bias Risk : Low  (score=0.043)
  No significant bias signals detected. Overall risk level: Low.
```

---

## 📈 Evaluation

The evaluation module uses FEVER-style test cases with known ground-truth labels (`"high"` = trustworthy, `"low"` = untrustworthy).

Metrics computed:
- **Precision** — Of all outputs labelled trustworthy, how many truly are?
- **Recall** — Of all truly trustworthy outputs, how many did we catch?
- **F1 Score** — Harmonic mean of precision and recall.
- **Accuracy** — Overall classification correctness.

Results are saved to `evaluation/results/evaluation_results.json`.

---

## ⚠️ Limitations

| Area | Limitation |
|------|-----------|
| **Knowledge Base** | Small sample KB (20 entries); production use needs a large, up-to-date corpus. |
| **Claim Extraction** | Heuristic sentence-level splitting; does not handle multi-sentence claims or implicit claims. |
| **NLI Model** | RoBERTa-MNLI may struggle with highly technical or domain-specific language. |
| **Bias Detection** | Keyword lists are not exhaustive; sentiment model was trained on movie reviews. |
| **Latency** | Running three transformer models sequentially on CPU can take 5–15 s per validation. |
| **Coverage** | Only English is supported. |

---

## 🔮 Future Scope

- **Live knowledge retrieval** — Integrate a web search API (e.g., Google, Bing) for real-time evidence.
- **Learned claim classifier** — Replace the heuristic confidence score with a fine-tuned model that distinguishes facts vs. opinions.
- **GPU acceleration** — Enable CUDA batching for <1 s latency.
- **Multi-language support** — Use multilingual Sentence-BERT and MNLI models.
- **User feedback loop** — Let users flag incorrect verdicts to improve the system over time.
- **API layer** — Expose the pipeline via FastAPI for programmatic integration.
- **Larger knowledge bases** — Use FAISS IVF or HNSW indexes for million-scale passage retrieval.
- **Toxicity model upgrade** — Replace keyword matching with Perspective API or a dedicated toxicity classifier.

---

## 📄 License

This project is provided for educational and research purposes.

---

*Built with ❤️ using Python, HuggingFace Transformers, Sentence-BERT, FAISS, spaCy, and Streamlit.*
