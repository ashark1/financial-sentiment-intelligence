import pandas as pd
import pytest

from financial_sentiment.data import load_dataset


def test_load_dataset_normalises_labels(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "text": [" Strong earnings ", "Guidance unchanged"],
            "label": ["Positive", " NEUTRAL "],
        }
    ).to_csv(path, index=False)

    frame = load_dataset(path)

    assert frame["text"].tolist() == ["Strong earnings", "Guidance unchanged"]
    assert frame["label"].tolist() == ["positive", "neutral"]


def test_load_dataset_rejects_unknown_label(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["Example"], "label": ["bullish"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Unsupported labels"):
        load_dataset(path)
