"""
app/streamlit_app.py
============================================================
Live demo UI - Canadian finance sentiment & intent classifier
============================================================

WHY CONFIDENCE BARS:
    Calibration analysis in Milestone 4 showed FinBERT's
    confidence scores are reasonably well-calibrated. Surfacing
    them lets users SEE when the model is unsure - the kind of
    feature real production classifiers expose.

WHY HTTP TO FASTAPI:
    Streamlit and FastAPI deploy as TWO services. The heavy
    PyTorch model lives only in the FastAPI container.

CONFIGURATION:
    Set API_URL env var to point at the FastAPI service.
    Defaults to localhost for local development.

RUN LOCALLY:
    Terminal A: uvicorn app.api:app --port 8000
    Terminal B: streamlit run app/streamlit_app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Canadian Finance Classifier",
    page_icon="CA",
    layout="wide",
)

st.title("Canadian Finance Sentiment & Intent Classifier")
st.markdown(
    "Fine-tuned FinBERT with two classification heads. "
    "Type a finance headline or paste a press release and the "
    "model predicts both **sentiment** (negative / neutral / "
    "positive) and **intent** (what the text is *doing*)."
)

with st.expander("About this model"):
    st.markdown(
        "- **Architecture:** FinBERT encoder (~110M params) "
        "with two task-specific heads sharing the same body.\n"
        "- **Training data:** ~36,000 finance examples across "
        "three label-quality tiers (PhraseBank expert labels, "
        "Twitter Financial News, weak-labeled Bank of Canada "
        "releases).\n"
        "- **Test performance:** macro F1 = 0.86 (sentiment), "
        "0.91 (intent), beating a TF-IDF + logistic regression "
        "baseline by +0.110 / +0.088."
    )

with st.sidebar:
    st.subheader("Service status")
    try:
        r = requests.get(f"{API_URL}/ready", timeout=3)
        if r.status_code == 200:
            st.success(f"API ready at {API_URL}")
        else:
            st.warning(f"API not ready ({r.status_code}). Loading?")
    except requests.RequestException:
        st.error(f"Cannot reach API at {API_URL}")
        st.caption(
            "If running locally, start the API in a separate "
            "terminal:  `uvicorn app.api:app --port 8000`"
        )

    st.divider()
    st.subheader("Try a sample")
    samples = [
        "Bank of Canada holds overnight rate at 5%, citing sticky core inflation.",
        "Shopify Q3 revenue rose 25% to $1.7B, beating analyst estimates.",
        "Manulife flags rising exposure to U.S. commercial real estate.",
        "TSX closed flat as energy gains offset bank weakness.",
        "RBC Capital upgrades Loblaw to Outperform with $185 target.",
        "Air Canada posts record quarterly loss as fuel costs surge.",
    ]
    chosen_sample = None
    for s in samples:
        if st.button(s, key=f"sample_{hash(s)}", use_container_width=True):
            chosen_sample = s

default_text = chosen_sample or ""
text = st.text_area(
    "Finance text to classify:",
    value=default_text,
    height=100,
    placeholder="Paste a headline, tweet, or press-release sentence...",
)

run = st.button("Classify", type="primary", use_container_width=True)


def render_head(title: str, head: dict, color: str):
    """Render one head's prediction as a styled card + bar chart."""
    label = head["label"]
    conf = head["confidence"]

    st.markdown(f"### {title}")
    st.markdown(
        f"<div style='padding:14px 18px; border-radius:10px; "
        f"background:{color}; color:white;'>"
        f"<div style='font-size:13px; opacity:0.85;'>Prediction</div>"
        f"<div style='font-size:26px; font-weight:600;'>{label}</div>"
        f"<div style='font-size:13px; opacity:0.85;'>"
        f"Confidence: {conf:.1%}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**All class scores**")
    scores = head["all_scores"]
    df = pd.DataFrame({
        "class": list(scores.keys()),
        "score": list(scores.values()),
    }).sort_values("score", ascending=True)
    st.bar_chart(df.set_index("class"), height=220)


if run:
    text_stripped = (text or "").strip()
    if len(text_stripped) < 3:
        st.warning("Please enter at least a few words.")
    else:
        with st.spinner("Classifying..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_stripped},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as exc:
                st.error(f"API call failed: {exc}")
                result = None

        if result:
            if result["cleaned_text"] != result["text"]:
                with st.expander("Preprocessed text (URLs, mentions stripped)"):
                    st.code(result["cleaned_text"])

            col1, col2 = st.columns(2)
            with col1:
                render_head("Sentiment", result["sentiment"], "#2563eb")
            with col2:
                render_head("Intent", result["intent"], "#16a34a")

            min_conf = min(
                result["sentiment"]["confidence"],
                result["intent"]["confidence"],
            )
            if min_conf < 0.55:
                st.info(
                    "**Low confidence** - the model is uncertain about this "
                    "one. The model's reliability diagram (see case study) "
                    "shows that low-confidence predictions are noticeably "
                    "less accurate than high-confidence ones, so treat "
                    "this result as a hint rather than a verdict."
                )

st.divider()
st.caption(
    "Model trained on Bank of Canada press releases, Financial "
    "PhraseBank, and Twitter Financial News data. Not investment advice."
)
