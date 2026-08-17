from financial_sentiment.model import build_pipeline


def test_pipeline_fits_and_predicts():
    texts = [
        "profit beats expectations",
        "revenue guidance raised",
        "results unchanged",
        "outlook maintained",
        "profit warning issued",
        "guidance cut sharply",
    ]
    labels = [
        "positive",
        "positive",
        "neutral",
        "neutral",
        "negative",
        "negative",
    ]

    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    prediction = pipeline.predict(["company raises profit outlook"])

    assert prediction.shape == (1,)
    assert prediction[0] in {"positive", "neutral", "negative"}
