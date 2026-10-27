"""Step 7: time forward and backward passes separately (median of 100 runs).

Writes results/timing.csv. Batch size 32 is the training batch size; the full
training set (1257 samples) is added to show how the picture changes when the
matrices are larger.
"""
import csv
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import nn  # noqa: E402

N_RUNS = 100
N_WARMUP = 10
WIDTHS = [16, 64]


def matmul_flops(width, batch):
    """Multiply-add FLOPs (2*m*k*n per matmul) in forward and backward."""
    fwd = 2 * batch * (64 * width + width * width + width * 10)
    # backward matmuls: dW3, da2, dW2, da1, dW1 (no gradient w.r.t. the input X)
    bwd = 2 * batch * (width * 10 + 10 * width + width * width + width * width + 64 * width)
    return fwd, bwd


def median_time_ms(fn):
    for _ in range(N_WARMUP):
        fn()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    os.makedirs("results", exist_ok=True)
    X_tr, _, _, _, Y_tr, _ = nn.load_data()

    rows = []
    for width in WIDTHS:
        params = nn.init_params(width, np.random.default_rng(nn.SEED), std=0.1)
        for batch in (32, len(X_tr)):
            Xb, Yb = X_tr[:batch], Y_tr[:batch]
            _, cache = nn.forward(params, Xb)

            fwd_ms = median_time_ms(lambda: nn.forward(params, Xb))
            bwd_ms = median_time_ms(lambda: nn.backward(params, cache, Yb))
            f_fl, b_fl = matmul_flops(width, batch)
            rows.append({
                "width": width,
                "batch": batch,
                "forward_ms": round(fwd_ms, 4),
                "backward_ms": round(bwd_ms, 4),
                "backward_over_forward": round(bwd_ms / fwd_ms, 2),
                "forward_matmul_flops": f_fl,
                "backward_matmul_flops": b_fl,
                "flops_ratio": round(b_fl / f_fl, 2),
            })

    with open("results/timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("width batch  fwd(ms)  bwd(ms)  bwd/fwd  fwd FLOPs  bwd FLOPs  FLOP ratio")
    for r in rows:
        print(f"{r['width']:<5} {r['batch']:<6} {r['forward_ms']:<8} {r['backward_ms']:<8} "
              f"{r['backward_over_forward']:<8} {r['forward_matmul_flops']:<10} "
              f"{r['backward_matmul_flops']:<10} {r['flops_ratio']}")


if __name__ == "__main__":
    main()
