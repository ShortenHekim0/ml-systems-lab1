import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS


def main():
    acc = pd.read_csv(RESULTS / "accuracy.csv")
    lat = pd.read_csv(RESULTS / "latency.csv")

    # Accuracy vs parameters
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(acc["params"], acc["test_accuracy"], s=70)
    for _, row in acc.iterrows():
        ax.annotate(row["model"], (row["params"], row["test_accuracy"]),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs parameters")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_params.png", dpi=150)
    plt.close(fig)

    # Accuracy vs latency (batch 1 and batch 128)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, batch in zip(axes, (1, 128)):
        sub = lat[lat["batch"] == batch].merge(acc, on="model")
        ax.scatter(sub["median_ms"], sub["test_accuracy"], s=70)
        for _, row in sub.iterrows():
            ax.annotate(row["model"], (row["median_ms"], row["test_accuracy"]),
                        textcoords="offset points", xytext=(6, 6))
        ax.set_xlabel(f"Median latency, batch {batch} (ms)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Accuracy vs latency")
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_latency.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
