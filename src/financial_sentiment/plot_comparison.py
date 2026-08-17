from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_results(input_path: str, output_path: str) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Visualisation support is not installed. Run: pip install -e .[viz]"
        ) from exc

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    baseline = [data["baseline"][name] for name in metrics]
    finbert = [data["finbert"][name] for name in metrics]

    x = list(range(len(metrics)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([value - width / 2 for value in x], baseline, width, label="TF-IDF + Logistic Regression")
    ax.bar([value + width / 2 for value in x], finbert, width, label="FinBERT")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Financial sentiment model comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("macro_", "").replace("_", " ").title() for name in metrics])
    ax.legend()
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline-vs-FinBERT evaluation metrics.")
    parser.add_argument("--input", default="artifacts/model_comparison.json")
    parser.add_argument("--output", default="docs/model_comparison.png")
    args = parser.parse_args()
    print(plot_results(args.input, args.output))


if __name__ == "__main__":
    main()
