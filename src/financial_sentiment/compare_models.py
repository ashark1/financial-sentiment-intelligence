from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .finbert import predict_finbert
from .real_data import load_financial_phrasebank


def metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def compare(model_path: str, limit: int | None = 300) -> dict:
    frame = load_financial_phrasebank("sentences_75agree")
    if limit:
        frame = frame.sample(min(limit, len(frame)), random_state=42)

    baseline = joblib.load(model_path)
    baseline_pred = baseline.predict(frame["text"])
    finbert_pred = [predict_finbert(text)["label"] for text in frame["text"]]

    return {
        "rows": int(len(frame)),
        "dataset": "Financial PhraseBank / sentences_75agree",
        "baseline": metrics(frame["label"], baseline_pred),
        "finbert": metrics(frame["label"], finbert_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline sentiment model with FinBERT.")
    parser.add_argument("--model", required=True, help="Path to baseline model.joblib")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output", default="artifacts/model_comparison.json")
    args = parser.parse_args()

    result = compare(args.model, args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
