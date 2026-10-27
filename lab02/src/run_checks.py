"""Steps 2-5: shapes, forward-pass sanity, initial loss, numerical gradient check.

Writes results/checks.txt and results/gradient_check.csv.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import nn  # noqa: E402

H = 1e-5
N_CHECK = 10
REL_TOL = 1e-6


def relative_error(a, b):
    return abs(a - b) / max(abs(a) + abs(b), 1e-12)


def gradient_check(params, X, Y, rng, label):
    """Nudge N_CHECK random weights (not biases) by +/-H, compare slopes."""
    _, cache = nn.forward(params, X)
    grads = nn.backward(params, cache, Y)

    names = ["W1", "W2", "W3"]
    rows = []
    seen = set()
    while len(rows) < N_CHECK:
        name = names[rng.integers(len(names))]
        i = rng.integers(params[name].shape[0])
        j = rng.integers(params[name].shape[1])
        # Skip duplicates and weights whose gradient is exactly zero (pixels
        # that are always 0, dead ReLU units): 0 vs 0 proves nothing.
        if (name, i, j) in seen or grads[name][i, j] == 0.0:
            continue
        seen.add((name, i, j))
        old = params[name][i, j]

        params[name][i, j] = old + H
        loss_plus = nn.cross_entropy(nn.forward(params, X)[0], Y)
        params[name][i, j] = old - H
        loss_minus = nn.cross_entropy(nn.forward(params, X)[0], Y)
        params[name][i, j] = old

        numeric = (loss_plus - loss_minus) / (2 * H)
        analytic = grads[name][i, j]
        rows.append({
            "setting": label,
            "weight": f"{name}[{i},{j}]",
            "analytic": analytic,
            "numeric": numeric,
            "rel_error": relative_error(analytic, numeric),
        })
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    X_tr, X_te, y_tr, y_te, Y_tr, Y_te = nn.load_data()
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"Train: {X_tr.shape}, Test: {X_te.shape}, labels one-hot: {Y_tr.shape}")

    all_rows = []
    for width in (16, 64):
        rng = np.random.default_rng(nn.SEED)
        params = nn.init_params(width, rng)
        log(f"\n=== W = {width} ===")
        for k, v in params.items():
            log(f"  {k}: shape {v.shape}")
        log(f"  total parameters: {nn.count_params(params)}")

        # Step 3: forward-pass checks
        p, _ = nn.forward(params, X_tr[:32])
        row_sum_err = np.abs(p.sum(axis=1) - 1.0).max()
        log(f"  output shape: {p.shape} (expected (32, 10))")
        log(f"  max |row sum - 1|: {row_sum_err:.2e} (must be < 1e-6)")
        assert p.shape == (32, 10) and row_sum_err < 1e-6

        # Step 4: untrained loss
        p_all, _ = nn.forward(params, X_tr)
        loss0 = nn.cross_entropy(p_all, Y_tr)
        log(f"  untrained loss: {loss0:.4f} (ln(10) = {np.log(10):.4f})")

        # Step 5: gradient check on a fixed batch of 64 training samples
        Xb, Yb = X_tr[:64], Y_tr[:64]
        check_rng = np.random.default_rng(0)
        rows = gradient_check(params, Xb, Yb, check_rng, f"W={width}, init N(0,0.01^2)")
        n_pass = sum(r["rel_error"] < REL_TOL for r in rows)
        log(f"  gradient check at the specified init: {n_pass}/{N_CHECK} weights pass")
        all_rows += rows

        # At std 0.01 the gradients are ~1e-5..1e-6, so float64 round-off in
        # (L(w+h) - L(w-h)) / 2h is already ~1e-6 relative: the check is
        # precision-limited there, not wrong. Re-check at std 0.1, where the
        # gradients are larger and the numerical slope is accurate.
        larger = nn.init_params(width, np.random.default_rng(nn.SEED), std=0.1)
        rows_larger = gradient_check(larger, Xb, Yb, np.random.default_rng(0),
                                     f"W={width}, init N(0,0.1^2)")
        n_pass_larger = sum(r["rel_error"] < REL_TOL for r in rows_larger)
        log(f"  gradient check at std 0.1:            {n_pass_larger}/{N_CHECK} weights pass")
        all_rows += rows_larger

    with open("results/gradient_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})

    with open("results/checks.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
