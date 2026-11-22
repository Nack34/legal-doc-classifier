"""Train the classifier and write the model plus an evaluation report.

Usage:  python -m classifier.train [--data data/escritos.csv] [--out artifacts]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from classifier.model import load_dataset, save, top_terms, train_and_evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def plot_confusion(labels: list[str], matrix: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if value > max(map(max, matrix)) / 2 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "escritos.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    texts, labels = load_dataset(args.data)
    model, result = train_and_evaluate(texts, labels)
    save(model, result, args.out)

    args.reports.mkdir(exist_ok=True)
    plot_confusion(result.labels, result.confusion, args.reports / "confusion_matrix.png")

    print(f"Documents: {len(texts)}  |  classes: {len(result.labels)}")
    print("Cross-validated macro F1 by model:")
    for name, score in sorted(result.cv_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {score:.3f}")
    print(f"Selected: {result.best_model}  |  test macro F1: {result.test_macro_f1:.3f}")
    for label, terms in top_terms(model).items():
        print(f"  {label:<22} {', '.join(terms)}")


if __name__ == "__main__":
    main()
