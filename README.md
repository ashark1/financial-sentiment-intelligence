# Financial Sentiment Intelligence

An end-to-end NLP project for classifying short financial headlines and market commentary as positive, neutral, or negative.

I built this as a production-style extension of my MSc work on financial news and stock-tweet sentiment analysis. The repository now includes an interpretable classical baseline, a FinBERT inference path, a real Financial PhraseBank data loader, model comparison tooling, an API, a Streamlit demo, Docker support, and automated tests.

## What the project covers

- CSV data validation and cleaning
- TF-IDF + logistic regression baseline
- FinBERT financial-sentiment inference
- Financial PhraseBank ingestion
- baseline-vs-FinBERT evaluation
- reproducible JSON metrics and comparison plot
- FastAPI prediction endpoint
- Streamlit demo interface
- Docker deployment path
- GitHub Actions test matrix

## Repository structure

```text
financial-sentiment-intelligence/
├── .github/workflows/ci.yml
├── data/sample/
├── docs/
├── src/financial_sentiment/
│   ├── api.py
│   ├── compare_models.py
│   ├── data.py
│   ├── demo_data.py
│   ├── evaluate.py
│   ├── finbert.py
│   ├── model.py
│   ├── plot_comparison.py
│   ├── predict.py
│   ├── real_data.py
│   └── train.py
├── tests/
├── Dockerfile
├── streamlit_app.py
├── pyproject.toml
└── README.md
```

## Install

For the lightweight baseline and tests:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

For the full project:

```bash
pip install -e .[all]
```

## 1. Train the baseline

The repository includes a small synthetic dataset so the pipeline can be run without downloading external data.

```bash
python -m financial_sentiment.train \
  --data data/sample/demo_financial_sentiment.csv \
  --model-dir artifacts
```

The baseline uses TF-IDF unigrams/bigrams and class-balanced logistic regression.

## 2. Load real financial text

The project can download the Financial PhraseBank dataset through Hugging Face Datasets and export it into the same `text,label` format used by the baseline.

```bash
python -m financial_sentiment.real_data \
  --config sentences_75agree \
  --output data/processed/financial_phrasebank.csv
```

The processed dataset is intentionally ignored by Git because it should be reproduced from source rather than committed as generated data.

## 3. Run FinBERT

```python
from financial_sentiment.finbert import predict_finbert

result = predict_finbert(
    "The company raised its full-year revenue guidance after strong demand."
)
print(result)
```

The transformer model is loaded lazily, so installing the basic package does not download model weights.

## 4. Compare the models

Train a baseline on a suitable labelled dataset, then run:

```bash
python -m financial_sentiment.compare_models \
  --model artifacts/model.joblib \
  --limit 300 \
  --output artifacts/model_comparison.json
```

This evaluates both models on the same Financial PhraseBank sample and records accuracy, macro precision, macro recall and macro F1.

Generate a chart from the saved metrics:

```bash
python -m financial_sentiment.plot_comparison \
  --input artifacts/model_comparison.json \
  --output docs/model_comparison.png
```

No benchmark numbers are hard-coded in this repository. Results should come from an actual run so the project does not present fabricated performance.

## 5. FastAPI service

Install API and transformer dependencies:

```bash
pip install -e .[transformers,api]
```

Start the service:

```bash
uvicorn financial_sentiment.api:app --reload
```

Endpoints:

```text
GET  /health
POST /predict
```

Example request body:

```json
{
  "text": "Profit beats expectations and management raises guidance."
}
```

## 6. Streamlit demo

```bash
pip install -e .[transformers,app]
streamlit run streamlit_app.py
```

The app provides a simple text box, predicted sentiment and confidence score.

## Docker

Build and run the API:

```bash
docker build -t financial-sentiment-intelligence .
docker run -p 8000:8000 financial-sentiment-intelligence
```

## Testing

```bash
pytest -q
```

The CI workflow runs the lightweight unit tests against Python 3.10, 3.11 and 3.12. Transformer weights are not downloaded during CI; this keeps routine checks fast and avoids coupling basic tests to an external model host.

## Data and model choices

**Financial PhraseBank** is used as the real financial-sentiment evaluation source. The project defaults to the `sentences_75agree` configuration, while the loader also supports the other published agreement thresholds.

**ProsusAI/finbert** provides the transformer comparison. It is a BERT-based model adapted to financial text and outputs positive, negative and neutral sentiment classes.

## Limitations

- sentiment is not the same as future stock-price movement
- the synthetic sample data is only a runnable fixture, not evidence of model quality
- a proper benchmark should use a fixed evaluation split and record the exact dataset/model versions
- model confidence is not automatically a calibrated probability
- long documents need chunking or another long-context strategy

This repository is for NLP engineering and portfolio demonstration. It is not investment advice or an automated trading system.

## Next improvements

- create a fixed train/validation/test protocol for Financial PhraseBank
- add calibration and confidence thresholds
- add batch inference and ticker/time aggregation
- persist model-comparison results from a reproducible benchmark run
- deploy the Streamlit or API demo publicly

## Skills demonstrated

Python, NLP, scikit-learn, Transformers, FinBERT, model evaluation, FastAPI, Streamlit, Docker, GitHub Actions, testing, packaging and reproducibility.

## License

MIT
