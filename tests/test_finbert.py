import pytest

from financial_sentiment.finbert import normalise_label


def test_normalise_named_labels():
    assert normalise_label("positive") == "positive"
    assert normalise_label("NEGATIVE") == "negative"
    assert normalise_label(" neutral ") == "neutral"


def test_normalise_legacy_label_ids():
    assert normalise_label("LABEL_0") == "positive"
    assert normalise_label("LABEL_1") == "negative"
    assert normalise_label("LABEL_2") == "neutral"


def test_normalise_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unexpected FinBERT label"):
        normalise_label("bullish")
