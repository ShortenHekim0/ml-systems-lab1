import csv
import statistics
import threading
import time

import joblib
import psutil

from baselines import RESULTS, get_data, get_models

MB = 1024 * 1024


def median_train_time(name, X, y):
    get_models()[name].fit(X, y)  # 1 warm-up
    times = []
    for _ in range(5):
        model = get_models()[name]
        start = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def median_latency_ms(model, sample):
    model.predict(sample)  # 1 warm-up
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def track_rss(fn):
    """Run fn while sampling process RSS. Returns (before_MB, peak_MB)."""
    proc = psutil.Process()
    before = proc.memory_info().rss
    peak = before
    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(0.001)

    thread = threading.Thread(target=sampler)
    thread.start()
    try:
        fn()
    finally:
        stop.set()
        thread.join()
        peak = max(peak, proc.memory_info().rss)
    return before / MB, peak / MB


def main():
    X_train, X_test, y_train, y_test = get_data()
    sample = X_test[:1]
    rows = []

    for name in get_models():
        train_s = median_train_time(name, X_train, y_train)

        model = get_models()[name]
        model.fit(X_train, y_train)
        latency_ms = median_latency_ms(model, sample)

        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = path.stat().st_size

        tr_before, tr_peak = track_rss(lambda: get_models()[name].fit(X_train, y_train))
        inf_before, inf_peak = track_rss(
            lambda: [model.predict(sample) for _ in range(100)]
        )

        row = {
            "model": name,
            "train_time_median_s": round(train_s, 4),
            "inference_latency_median_ms": round(latency_ms, 4),
            "model_size_bytes": size_bytes,
            "model_size_kb": round(size_bytes / 1024, 2),
            "train_rss_before_mb": round(tr_before, 1),
            "train_rss_peak_mb": round(tr_peak, 1),
            "train_rss_delta_mb": round(tr_peak - tr_before, 1),
            "infer_rss_before_mb": round(inf_before, 1),
            "infer_rss_peak_mb": round(inf_peak, 1),
            "infer_rss_delta_mb": round(inf_peak - inf_before, 1),
        }
        rows.append(row)
        print(row)

    with open(RESULTS / "system_costs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()