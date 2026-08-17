from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .finbert import predict_finbert

app = FastAPI(
    title="Financial Sentiment Intelligence API",
    version="0.2.0",
    description="Classify short financial text with FinBERT.",
)


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: SentimentRequest) -> dict:
    try:
        return predict_finbert(request.text)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
