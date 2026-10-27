"""Two-hidden-layer neural network from scratch (NumPy only).

Architecture: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax
Loss: mean cross-entropy over the batch.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
N_IN, N_OUT = 64, 10


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def one_hot(y, n_classes=N_OUT):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def load_data(seed=SEED):
    """load_digits, inputs / 16, 70/30 stratified split, one-hot labels."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te, one_hot(y_tr), one_hot(y_te)


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
def init_params(width, rng, std=0.01):
    """Weights ~ N(0, std^2), biases = 0.

    Shapes: W1 (64,W), W2 (W,W), W3 (W,10); b1 (W,), b2 (W,), b3 (10,).
    """
    return {
        "W1": rng.normal(0.0, std, (N_IN, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0.0, std, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0.0, std, (width, N_OUT)),
        "b3": np.zeros(N_OUT),
    }


def count_params(params):
    return sum(p.size for p in params.values())


# ----------------------------------------------------------------------------
# Forward pass, layer by layer
# ----------------------------------------------------------------------------
def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    """Return (probabilities, cache). cache holds what backward needs."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    p = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "p": p}
    return p, cache


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def cross_entropy(p, Y):
    """Mean cross-entropy: -(1/N) * sum_n sum_k Y_nk * log(p_nk)."""
    return -np.mean(np.sum(Y * np.log(np.clip(p, 1e-300, 1.0)), axis=1))


# ----------------------------------------------------------------------------
# Backward pass (chain rule, derived by hand)
# ----------------------------------------------------------------------------
def backward(params, cache, Y):
    """Gradients of the mean cross-entropy w.r.t. every parameter.

    softmax + cross-entropy:  dL/dz3 = (p - Y) / N
    linear layer  z = a W + b: dW = a^T dz,  db = sum_rows(dz),  da = dz W^T
    ReLU:                      dz = da * (z > 0)
    """
    n = cache["X"].shape[0]
    dz3 = (cache["p"] - Y) / n
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def predict(params, X):
    p, _ = forward(params, X)
    return p.argmax(axis=1)


def accuracy(params, X, y):
    return float(np.mean(predict(params, X) == y))


def train(width, lr, epochs=50, batch_size=32, seed=SEED, data=None, verbose=True,
          init_std=0.01):
    """Mini-batch SGD. Returns (params, history list of dicts)."""
    if data is None:
        data = load_data(seed)
    X_tr, X_te, y_tr, y_te, Y_tr, Y_te = data
    rng = np.random.default_rng(seed)
    params = init_params(width, rng, std=init_std)
    history = []
    n = X_tr.shape[0]

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)  # shuffle every epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            _, cache = forward(params, X_tr[idx])
            grads = backward(params, cache, Y_tr[idx])
            for k in params:
                params[k] -= lr * grads[k]

        p_tr, _ = forward(params, X_tr)
        p_te, _ = forward(params, X_te)
        row = {
            "epoch": epoch,
            "train_loss": cross_entropy(p_tr, Y_tr),
            "test_loss": cross_entropy(p_te, Y_te),
            "train_acc": float(np.mean(p_tr.argmax(1) == y_tr)),
            "test_acc": float(np.mean(p_te.argmax(1) == y_te)),
        }
        history.append(row)
        if verbose:
            print(
                f"W={width:<3} lr={lr:<6} epoch {epoch:>2}/{epochs}  "
                f"train_loss={row['train_loss']:.4f}  test_loss={row['test_loss']:.4f}  "
                f"train_acc={row['train_acc']:.4f}  test_acc={row['test_acc']:.4f}"
            )
    return params, history
