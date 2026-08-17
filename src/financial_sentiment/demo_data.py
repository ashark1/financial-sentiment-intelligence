from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROWS = [
    ("Company raises full-year revenue guidance after strong demand", "positive"),
    ("Quarterly profit beats analyst expectations", "positive"),
    ("Shares rise after the board approves a new buyback programme", "positive"),
    ("Operating margin improves despite higher input costs", "positive"),
    ("New contract adds significant recurring revenue", "positive"),
    ("The firm reports record quarterly free cash flow", "positive"),
    ("Management expects stronger growth next quarter", "positive"),
    ("Dividend increased following better-than-expected earnings", "positive"),
    ("Results were broadly in line with market expectations", "neutral"),
    ("The company maintained its previous full-year outlook", "neutral"),
    ("Shares were little changed after the announcement", "neutral"),
    ("Revenue was flat compared with the previous quarter", "neutral"),
    ("The board said it continues to review strategic options", "neutral"),
    ("The company announced a scheduled leadership transition", "neutral"),
    ("Trading remained stable during the period", "neutral"),
    ("Management reiterated existing guidance", "neutral"),
    ("Profit warning sends shares sharply lower", "negative"),
    ("Company cuts full-year revenue guidance", "negative"),
    ("Quarterly loss widens as costs increase", "negative"),
    ("Regulatory investigation creates uncertainty for investors", "negative"),
    ("Demand weakens across key markets", "negative"),
    ("The firm suspends its dividend after a cash flow decline", "negative"),
    ("Margins fall because of higher operating expenses", "negative"),
    ("Management expects lower earnings next quarter", "negative"),
]


def write_demo_data(output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS, columns=["text", "label"]).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic demo dataset.")
    parser.add_argument(
        "--output",
        default="data/sample/demo_financial_sentiment.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()
    path = write_demo_data(args.output)
    print(path)


if __name__ == "__main__":
    main()
