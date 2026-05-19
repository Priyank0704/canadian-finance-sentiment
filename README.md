# Canadian Finance — Sentiment & Intent Classifier

Fine-tuned **FinBERT** with a multi-task architecture: one shared transformer encoder, two classification heads predicting **sentiment** (negative / neutral / positive) and **intent** (what the text is *doing* — reporting performance, signaling policy, warning of risk, etc.) on Canadian financial text.

### 🚀 [Try the live demo →](https://huggingface.co/spaces/Priyank0704/canadian-finance-classifier)

Hosted on HuggingFace Spaces. The first load takes ~15 seconds while the container wakes up; predictions after that are ~50–200ms each on CPU.

---

## Headline results

| Head | Baseline (TF-IDF + Logistic Regression) | FinBERT (fine-tuned) | Δ |
|---|---:|---:|---:|
| **Sentiment** | 0.748 | **0.858** | **+0.110** |
| **Intent**    | 0.824 | **0.912** | **+0.088** |

Macro F1 on a held-out test set of 5,455 examples. The largest per-class gains came on the rare classes (`negative` sentiment, `analyst_question` intent) where the bag-of-words baseline has no semantic anchor to fall back on.

---

## What makes this project worth a look

Most public finance-sentiment models are sentiment-only and trained on US data. This one differs on three deliberate axes:

- **Two heads on one encoder.** Trained with *partial supervision* — many rows have only one of the two labels (PhraseBank has no intent; the Twitter intent dataset has no sentiment), so the loss masks missing labels with `-100` and `cross_entropy(ignore_index=-100)` skips them. The shared encoder still benefits from every row regardless of which label is present.
- **Canadian context.** Training data includes Bank of Canada press releases and Canadian-market financial news alongside the more standard PhraseBank + Twitter Financial News, so the model sees the language patterns that matter for the local market.
- **Honest evaluation.** Three-tier labeling strategy (expert / curated / weak), with the test set drawn *only* from tier 1 and tier 2 so test scores aren't an evaluation of my own weak-labeling rules. A manual 100-row error audit categorizes the remaining mistakes into ambiguity, label noise, model bias, and missing context.

---

## Architecture

```
                          ┌─────────────────┐
                          │  Sentiment head │  →  3 classes
                          │   (Linear)      │
                ┌────────►└─────────────────┘
                │
   Text ──► FinBERT encoder (~110M params)
                │
                │         ┌─────────────────┐
                └────────►│   Intent head   │  →  5 classes
                          │   (Linear)      │
                          └─────────────────┘
```

The encoder is `yiyanghkust/finbert-tone` — BERT pre-trained on financial text. Two new linear heads sit on top of the shared `[CLS]` representation. The whole stack is fine-tuned end-to-end with class-weighted cross-entropy on both heads, with loss masking for the partial-supervision rows.

---

## Dataset

A **three-tier labeling strategy** combines ~36,000 examples:

| Tier | Source | Rows | Sentiment | Intent | Quality |
|---|---|---:|:---:|:---:|---|
| 1 | Financial PhraseBank (75%-agree) | ~3,400 | ✅ | — | Expert-annotated |
| 2 | Twitter Financial News sentiment | ~12,000 | ✅ | — | Crowd-curated |
| 2 | Twitter Financial News topic    | ~21,000 | — | ✅ | Crowd-curated (20 topics → my 5-class intent taxonomy) |
| 3 | Bank of Canada press releases   | ~40 | ✅ | ✅ | Rule-based weak supervision |

A 40-row manual audit of the weak labels gave **85% sentiment accuracy and 97.5% intent accuracy**, confirming the rule-based bootstrap was reliable enough to bring into training as low-weight signal. Train/val/test splits stratify by source and tier so the test set holds only tier-1 and tier-2 labels.

---

## Tech stack

- **Modeling:** PyTorch, HuggingFace Transformers, FinBERT, scikit-learn (baseline), BERTopic (taxonomy validation)
- **Experiment tracking:** Weights & Biases
- **Serving:** FastAPI (Pydantic v2 schemas, `/health` + `/ready` probes, batch endpoint), Streamlit (interactive demo with confidence bars and low-confidence flagging)
- **Deployment:** Docker, HuggingFace Spaces, Git LFS for model weights
- **Testing:** pytest with FastAPI TestClient

---

## Repository structure

```
canadian-finance-sentiment/
├── app/
│   ├── api.py                  # FastAPI inference service
│   ├── streamlit_app.py        # Live demo UI
│   └── entrypoint.sh           # Supervisor for both services (used by Docker)
├── src/
│   ├── collection/             # Data scrapers (PhraseBank, BoC, news)
│   ├── labeling/               # 3-tier label pipeline + BERTopic + audit tools
│   ├── features/               # Train/val/test splits with stratification
│   ├── training/               # Baseline + multi-task FinBERT training
│   ├── evaluation/             # Confusion matrices, error taxonomy, calibration
│   ├── inference/              # Preprocessing + predictor singleton
│   └── utils/                  # Central config
├── tests/                      # pytest smoke tests for the API
├── reports/
│   ├── CASE_STUDY.md           # Full writeup with figures and findings
│   ├── eval/                   # Metrics, per-class deltas, audit summaries
│   └── figures/                # Confusion matrices, calibration plot, etc.
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Reproducing the results locally

Requires Python 3.11, ~4 GB free RAM, ~5 GB free disk. Training is CPU-only; no GPU needed.

```bash
git clone https://github.com/Priyank0704/canadian-finance-sentiment.git
cd canadian-finance-sentiment

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

### Full pipeline (run from project root, in order):

```bash
# Milestone 1 — data collection
python -m src.collection.load_phrasebank
python -m src.collection.scrape_boc
python -m src.collection.scrape_news
python -m src.collection.build_master

# Milestone 2 — labeling
python -m src.labeling.load_twitter_finance
python -m src.labeling.weak_labeler
python -m src.labeling.explore_bertopic
python -m src.labeling.build_labeled_master

# Milestone 3 — training (the long one — 2–5 hours on CPU)
python -m src.features.build_splits
python -m src.training.train_baseline
python -m src.training.train_finbert --fast    # 5-min smoke test first
python -m src.training.train_finbert           # full run
python -m src.evaluation.evaluate_finbert

# Milestone 4 — analysis
python -m src.evaluation.confusion_matrices
python -m src.evaluation.per_class_delta
python -m src.evaluation.slice_and_calibration
python -m src.evaluation.error_taxonomy sample # fill in the CSV manually
python -m src.evaluation.error_taxonomy score
python -m src.evaluation.generate_case_study
```

### Run the demo locally:

```bash
# Terminal A — start the API
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal B — start the UI (it calls the API)
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

### Run the tests:

```bash
pytest tests/ -v
```

---

## Case study

The full writeup with figures, per-class breakdowns, error taxonomy results, and calibration analysis lives in [`reports/CASE_STUDY.md`](reports/CASE_STUDY.md).

---

## Limitations

- Trained primarily on English financial text — output quality drops on non-English text or non-finance domains.
- The 5-class intent taxonomy was designed for this project. Originally specified as 6 classes; `forward_guidance` was merged into `market_commentary` after labeling revealed only 20 rows in that class — a defensible mid-project taxonomy revision.
- The model is calibrated (Expected Calibration Error on the sentiment head is reported in the case study), but the demo still surfaces a low-confidence flag whenever either head's confidence is below 55%.
- **Not investment advice.**

---

## License

Apache 2.0.

## Acknowledgements

- [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) — Malo et al., 2014
- [Twitter Financial News datasets](https://huggingface.co/zeroshot) — zeroshot
- [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone) — Yang et al., 2020
- Bank of Canada press release archive

---

Built as part of a 6-project ML portfolio. Find me on [GitHub](https://github.com/Priyank0704) · [HuggingFace](https://huggingface.co/Priyank0704).
