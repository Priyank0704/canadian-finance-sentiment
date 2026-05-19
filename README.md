---
title: Canadian Finance Classifier
emoji: "CA"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Sentiment + intent classifier for Canadian financial text
---

# Canadian Finance Sentiment & Intent Classifier

Fine-tuned FinBERT with two classification heads on a shared transformer body.
Predicts both **sentiment** (negative / neutral / positive) and **intent**
(what the text is *doing* - reporting performance, signaling policy,
warning of risk, etc.) on Canadian financial text.

## Highlights

- **Architecture:** FinBERT encoder (~110M params) + two task-specific heads
  trained with partial supervision (rows with only one label still contribute
  gradient through the shared encoder).
- **Training data:** ~36,000 labeled examples across three quality tiers
  (PhraseBank expert labels, Twitter Financial News, weak-labeled Bank of
  Canada releases).
- **Test performance:** macro F1 of **0.86** (sentiment) and **0.91** (intent),
  beating a TF-IDF + logistic regression baseline by **+0.110** and **+0.088**.
- **Source code:** [GitHub repo](https://github.com/Priyank0704/canadian-finance-sentiment)

## How to use

Type or paste a finance headline into the box and click "Classify." The model
returns both sentiment and intent predictions with confidence scores. When
confidence is low (under ~55%), the UI flags the prediction as uncertain.

## How it works under the hood

This Space runs two services in one container:

- **FastAPI** on `localhost:8000` - the inference engine (model + tokenization)
- **Streamlit** on port 7860 - the UI you see now, which calls FastAPI

The model loads once at container startup. After that, predictions take
~50-200ms each on CPU.

## Limitations

- Trained primarily on English financial text. Output quality drops on
  non-English text or non-finance domains.
- Predicts only the 5-class intent taxonomy designed for this project
  (`market_commentary`, `performance_report`, `policy_signal`, `risk_warning`,
  `analyst_question`).
- Not investment advice.
