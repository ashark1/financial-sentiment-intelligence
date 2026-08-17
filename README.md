# Financial Sentiment Intelligence

A small end-to-end NLP project for classifying financial headlines and short market commentary as positive, neutral, or negative.

I built this as a production-style extension of my MSc work on financial news and stock-tweet sentiment analysis. The aim here is not to chase a headline accuracy number, but to show a clean workflow that can be reproduced, tested, and extended.

## What it does

- loads labelled financial text from CSV
- cleans and validates the input data
- trains a TF-IDF + logistic regression baseline
- reports accuracy and macro precision/recall/F1
- saves the trained pipeline and metadata
- runs predictions from the command line
- includes tests for preprocessing and model behaviour

## Why this project

Financial sentiment is a useful NLP problem because the same word can mean something different in a market context. A phrase such as "beats estimates" is usually positive, while "guidance cut" is usually negative. A useful system therefore needs more than simple positive/negative word counting.

This repository starts with an interpretable baseline. The next iteration will compare it with a transformer model and add time-based aggregation for market sentiment monitoring.

## Project structure

```text
financial-sentiment-intelligence/
├── data/
│   └── sample/
├── docs/
├── notebooks/
├── src/
│   └── financial_sentiment/
├── tests/
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Expected input format

A CSV file with two columns:

```csv
text,label
"Company raises full-year guidance",positive
"Shares are unchanged after the announcement",neutral
"Profit warning sends shares lower",negative
```

Accepted labels are `positive`, `neutral`, and `negative`.

## Quick start

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Generate a small demo dataset:

```bash
python -m financial_sentiment.demo_data --output data/sample/demo_financial_sentiment.csv
```

Train the model:

```bash
python -m financial_sentiment.train \
  --data data/sample/demo_financial_sentiment.csv \
  --model-dir artifacts
```

Predict a new sentence:

```bash
python -m financial_sentiment.predict \
  --model artifacts/model.joblib \
  --text "The company raised its revenue outlook after strong demand"
```

Run the tests:

```bash
pytest
```

## Modelling approach

The current baseline uses:

- lowercase text normalisation
- TF-IDF features with unigrams and bigrams
- logistic regression with class balancing
- stratified train/test split
- macro-averaged metrics so that each sentiment class matters

The training script saves both the fitted scikit-learn pipeline and a JSON metadata file with the evaluation results.

## Limitations

This first version is deliberately small. It does not claim to predict stock-price movement, and sentiment should not be treated as an investment signal on its own. The included demo data is synthetic and exists only to make the repository runnable.

Planned improvements:

1. evaluate on a real public financial-sentiment dataset
2. add a FinBERT comparison
3. introduce calibration and confidence thresholds
4. aggregate sentiment by ticker and time window
5. expose predictions through a lightweight API
6. add a dashboard for model and sentiment monitoring

## Skills demonstrated

Python, NLP, scikit-learn, TF-IDF, classification, model evaluation, packaging, testing, reproducibility, and basic MLOps practices.

## License

MIT
