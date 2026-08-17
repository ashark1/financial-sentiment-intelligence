from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "label_0": "positive",
    "label_1": "negative",
    "label_2": "neutral",
}


def normalise_label(label: str) -> str:
    key = label.strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unexpected FinBERT label: {label}")
    return LABEL_MAP[key]


@lru_cache(maxsize=1)
def get_pipeline():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Transformer support is not installed. Run: pip install -e .[transformers]"
        ) from exc

    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
    )


def predict_finbert(text: str) -> dict[str, float | str]:
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    result = get_pipeline()(text.strip())[0]
    return {
        "text": text.strip(),
        "label": normalise_label(str(result["label"])),
        "confidence": float(result["score"]),
        "model": MODEL_NAME,
    }
