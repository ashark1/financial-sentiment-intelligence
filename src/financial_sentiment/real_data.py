from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_financial_phrasebank(config: str = "sentences_75agree") -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dataset support is not installed. Run: pip install -e .[data]"
        ) from exc

    dataset = load_dataset("takala/financial_phrasebank", config)
    split = dataset["train"]

    labels = split["label"]
    if labels and not isinstance(labels[0], str):
        feature = split.features["label"]
        labels = [feature.int2str(int(value)) for value in labels]

    frame = pd.DataFrame({"text": split["sentence"], "label": labels})
    frame["label"] = frame["label"].astype(str).str.lower()
    return frame.dropna().drop_duplicates().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Financial PhraseBank and export it as text/label CSV."
    )
    parser.add_argument(
        "--config",
        default="sentences_75agree",
        choices=[
            "sentences_50agree",
            "sentences_66agree",
            "sentences_75agree",
            "sentences_allagree",
        ],
    )
    parser.add_argument(
        "--output",
        default="data/processed/financial_phrasebank.csv",
    )
    args = parser.parse_args()

    frame = load_financial_phrasebank(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
