from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"text", "label"}
VALID_LABELS = {"positive", "neutral", "negative"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.loc[:, ["text", "label"]].copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame = frame[(frame["text"] != "") & frame["label"].notna()]
    frame = frame.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    unknown = set(frame["label"].unique()).difference(VALID_LABELS)
    if unknown:
        raise ValueError(f"Unsupported labels found: {sorted(unknown)}")

    if frame.empty:
        raise ValueError("Dataset is empty after validation.")

    return frame
