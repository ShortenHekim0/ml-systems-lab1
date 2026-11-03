import csv
import time

import torch
from torch import nn

from common import MODEL_NAMES, RESULTS, build_model, get_loaders, set_seeds

EPOCHS = 5
LR = 1e-3


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += loss_fn(out, y).item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            count += y.size(0)
    return total_loss / count, correct / count


def main():
    loss_fn = nn.CrossEntropyLoss()
    summary_rows = []
    log_rows = []

    for name in MODEL_NAMES:
        set_seeds()
        train_loader, val_loader, test_loader = get_loaders()
        model = build_model(name)
        params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running, seen = 0.0, 0
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * y.size(0)
                seen += y.size(0)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn)
            log_rows.append(
                {
                    "model": name,
                    "epoch": epoch,
                    "train_loss": round(running / seen, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                }
            )
            print(name, "epoch", epoch, "val acc", round(val_acc, 4))
        train_seconds = time.perf_counter() - start

        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        print(name, "test accuracy", f"{test_acc:.4f}")
        summary_rows.append(
            {
                "model": name,
                "params": params,
                "test_accuracy": f"{test_acc:.4f}",
                "train_seconds": round(train_seconds, 1),
            }
        )

    with open(RESULTS / "accuracy.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "params", "test_accuracy", "train_seconds"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(RESULTS / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "epoch", "train_loss", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(log_rows)


if __name__ == "__main__":
    main()
