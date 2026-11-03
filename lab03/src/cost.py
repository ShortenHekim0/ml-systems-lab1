import csv
import statistics
import time

import torch
from thop import profile
from torch import nn
from torchinfo import summary

from common import MODEL_NAMES, RESULTS, build_model, set_seeds

INPUT_SHAPE = (1, 1, 28, 28)


def set_fastpath(enabled):
    """The fused transformer fast path skips submodule calls, so thop would miss layers."""
    try:
        torch.backends.mha.set_fastpath_enabled(enabled)
    except AttributeError:
        pass


def thop_macs(module, example):
    module.eval()
    macs, params = profile(module, inputs=(example,), verbose=False)
    return int(macs), int(params)


def median_latency_ms(model, batch, runs):
    x = torch.randn(batch, 1, 28, 28)
    times = []
    with torch.no_grad():
        model(x)  # 1 warm-up
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    set_seeds()

    # 1) Parameter counts with torchinfo
    param_rows = []
    for name in MODEL_NAMES:
        model = build_model(name)
        stats = summary(model, input_size=INPUT_SHAPE, verbose=0)
        (RESULTS / f"torchinfo_{name}.txt").write_text(str(stats), encoding="utf-8")
        param_rows.append({"model": name, "params": stats.total_params})
        print(name, "params:", stats.total_params)
    with open(RESULTS / "param_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "params"])
        writer.writeheader()
        writer.writerows(param_rows)

    # 2) thop: whole models and the single layers used for the hand calculation
    set_fastpath(False)
    thop_rows = []
    for name in MODEL_NAMES:
        macs, _ = thop_macs(build_model(name), torch.randn(*INPUT_SHAPE))
        thop_rows.append(("whole model", name, macs))

    single_layers = [
        ("single layer", "MLP Linear 784->256", nn.Linear(784, 256), torch.randn(1, 784)),
        (
            "single layer",
            "CNN Conv2d 16->32 (input 14x14)",
            nn.Conv2d(16, 32, 3, padding=1),
            torch.randn(1, 16, 14, 14),
        ),
        (
            "single layer",
            "Attention QKV projection (seq 16)",
            nn.Linear(64, 192),
            torch.randn(1, 16, 64),
        ),
    ]
    for scope, item, module, example in single_layers:
        macs, _ = thop_macs(module, example)
        thop_rows.append((scope, item, macs))
    set_fastpath(True)

    with open(RESULTS / "thop_flops.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "item", "thop_macs", "thop_macs_times_2"])
        for scope, item, macs in thop_rows:
            writer.writerow([scope, item, macs, macs * 2])
            print(scope, "|", item, "| MACs:", macs)

    # 3) Latency: batch 1 (median of 100 runs) and batch 128 (median of 20 runs)
    latency_rows = []
    for name in MODEL_NAMES:
        model = build_model(name)
        model.eval()
        for batch, runs in ((1, 100), (128, 20)):
            ms = median_latency_ms(model, batch, runs)
            latency_rows.append(
                {
                    "model": name,
                    "batch": batch,
                    "runs": runs,
                    "threads": torch.get_num_threads(),
                    "median_ms": round(ms, 4),
                }
            )
            print(name, "batch", batch, "median ms:", round(ms, 4))
    with open(RESULTS / "latency.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "batch", "runs", "threads", "median_ms"]
        )
        writer.writeheader()
        writer.writerows(latency_rows)


if __name__ == "__main__":
    main()
