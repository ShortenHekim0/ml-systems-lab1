"""Step 6: train all (init std, width, learning rate) combinations and plot loss curves.

"N(0, 0.01)" can be read two ways: std = 0.01 (literal) or variance = 0.01
(std = 0.1). Both are run so the report can compare them.

Writes results/training_log.csv, results/final_accuracy.csv,
results/loss_curves.png (std 0.01) and results/loss_curves_std0.1.png (std 0.1).
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import nn  # noqa: E402

WIDTHS = [16, 64]
LEARNING_RATES = [0.001, 0.01, 0.1]
INIT_STDS = [0.01, 0.1]
EPOCHS = 50
BATCH_SIZE = 32


def plot(histories, std, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, width in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            hist = histories[(std, width, lr)]
            ax.plot([r["epoch"] for r in hist], [r["train_loss"] for r in hist],
                    label=f"lr={lr}")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Training cross-entropy loss")
    fig.suptitle(f"Training loss, weights init N(0, std={std}), batch size {BATCH_SIZE}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs("results", exist_ok=True)
    data = nn.load_data()

    histories = {}
    for std in INIT_STDS:
        for width in WIDTHS:
            for lr in LEARNING_RATES:
                _, hist = nn.train(width, lr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                   data=data, init_std=std)
                histories[(std, width, lr)] = hist

    with open("results/training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["init_std", "width", "lr", "epoch", "train_loss", "test_loss",
                    "train_acc", "test_acc"])
        for (std, width, lr), hist in histories.items():
            for r in hist:
                w.writerow([std, width, lr, r["epoch"], f"{r['train_loss']:.6f}",
                            f"{r['test_loss']:.6f}", f"{r['train_acc']:.4f}",
                            f"{r['test_acc']:.4f}"])

    with open("results/final_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["init_std", "width", "lr", "final_train_loss", "final_test_loss",
                    "final_test_accuracy"])
        for (std, width, lr), hist in histories.items():
            last = hist[-1]
            w.writerow([std, width, lr, f"{last['train_loss']:.4f}",
                        f"{last['test_loss']:.4f}", f"{last['test_acc']:.4f}"])

    plot(histories, 0.01, "results/loss_curves.png")
    plot(histories, 0.1, "results/loss_curves_std0.1.png")

    print("\ninit_std  width  lr      final train loss  final test accuracy")
    for (std, width, lr), hist in histories.items():
        last = hist[-1]
        print(f"{std:<9} {width:<6} {lr:<7} {last['train_loss']:<17.4f} {last['test_acc']:.4f}")


if __name__ == "__main__":
    main()
