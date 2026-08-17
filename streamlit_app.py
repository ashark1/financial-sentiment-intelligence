from __future__ import annotations

import streamlit as st

from financial_sentiment.finbert import predict_finbert

st.set_page_config(page_title="Financial Sentiment Intelligence", page_icon="📈")
st.title("Financial Sentiment Intelligence")
st.caption("A lightweight FinBERT demo for short financial headlines and commentary.")

text = st.text_area(
    "Financial text",
    placeholder="Example: The company raised its full-year revenue guidance after strong demand.",
    height=140,
)

if st.button("Analyse sentiment", type="primary"):
    if not text.strip():
        st.warning("Enter some financial text first.")
    else:
        with st.spinner("Running FinBERT..."):
            try:
                result = predict_finbert(text)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.metric("Sentiment", str(result["label"]).title())
                st.progress(float(result["confidence"]))
                st.caption(f"Confidence: {float(result['confidence']):.1%}")
                st.caption(f"Model: {result['model']}")

st.divider()
st.caption(
    "This project demonstrates NLP engineering. Sentiment output is not investment advice or a prediction of market returns."
)
