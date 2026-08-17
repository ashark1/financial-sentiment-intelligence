from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .evaluate import classification_metrics
from .model import build_pipeline


def train(data_path: str | Path, model_dir: str | Path) -> dict:
    frame = load_dataset(data_path)

    if frame["label"].nunique() < 2:
        raise ValueError("Training requires at least two sentiment classes.")

    counts = frame["label"].value_counts()
    if counts.min() < 2:
        raise ValueError("Each class needs at least two examples for a stratified split.")

    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["label"],
        test_size=0.25,
        random_state=42,
        stratify=frame["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = classification_metrics(y_test, predictions)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "model": "tfidf_logistic_regression",
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "labels": sorted(frame["label"].unique().tolist()),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a financial sentiment classifier.")
    parser.add_argument("--data", required=True, help="Path to labelled CSV data.")
    parser.add_argument("--model-dir", default="artifacts", help="Directory for model output.")
    args = parser.parse_args()

    metadata = train(args.data, args.model_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
