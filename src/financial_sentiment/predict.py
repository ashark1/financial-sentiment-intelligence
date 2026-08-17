from __future__ import annotations

import argparse
import json

import joblib


def predict_text(model_path: str, text: str) -> dict:
    model = joblib.load(model_path)
    label = model.predict([text])[0]

    result = {"text": text, "label": str(label)}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_
        result["probabilities"] = {
            str(cls): float(prob) for cls, prob in zip(classes, probabilities)
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for one text input.")
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--text", required=True, help="Financial text to classify")
    args = parser.parse_args()

    print(json.dumps(predict_text(args.model, args.text), indent=2))


if __name__ == "__main__":
    main()
