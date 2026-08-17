# Model card

## Model

TF-IDF vectorisation followed by multinomial logistic regression.

## Intended use

Educational and portfolio use for classifying short pieces of financial text into positive, neutral, and negative sentiment.

## Not intended for

- investment advice
- automated trading
- estimating future share-price movement
- high-stakes financial decisions

## Data

The repository includes a small synthetic dataset purely to demonstrate the training pipeline. It is not suitable for claiming real-world model performance.

## Evaluation

The training command creates `artifacts/metadata.json` with accuracy and macro-averaged precision, recall, and F1 on the held-out split.

## Known limitations

The baseline does not explicitly model financial entities, sarcasm, long context, temporal information, or domain-specific transformer representations.
