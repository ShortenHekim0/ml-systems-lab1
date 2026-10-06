# Lab 1: Environment and first system measurements

## 1. Goal
The goal of this lab is to set up a reproducible, pinned Python environment and use it to train two baseline models (logistic regression and random forest) on the Breast Cancer Wisconsin dataset. I then measure their real system costs: training time, single-sample latency, model size, and memory. Finally, I use these measurements to decide which deployment targets (Cloud, Edge, Mobile, TinyML) each model could realistically run on.

## 2. Method
- Environment: Python 3.11.8 on Windows 10, pinned packages (see the README for the pyarrow note). Versions: results/versions.txt
- Data: Breast Cancer Wisconsin, 70%/30% stratified split, random_state=42, all seeds set to 42
- Models: LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=100)
- Training time: median of 1 warm-up + 5 runs
- Single-sample inference: median of 1 warm-up + 100 repetitions
- Model size: joblib.dump, in bytes and KB (the two models were saved to separate files)
- Memory: process RSS measured with psutil, during training and inference
- Code: src/baselines.py, src/measure_costs.py

## 3. Results

| Measurement | LogisticRegression | RandomForest |
|---|---|---|
| Test accuracy | 0.9415 | 0.9357 |
| Training time (median, s) | 0.1551 | 0.1610 |
| Single-sample latency (median, ms) | 0.0524 | 2.0155 |
| Model size (bytes) | 1055 | 290889 |
| Model size (KB) | 1.03 | 284.07 |
| Process RSS peak, training (MB) | 226.4 | 227.9 |
| Process RSS peak, inference (MB) | 226.5 | 227.9 |

Notes:
- ConvergenceWarning: LogisticRegression raised a ConvergenceWarning. The features are not scaled, so the lbfgs solver converges slowly. Even with max_iter=1000 (the default is 100), the fitted model reports n_iter_ = 1000, which means it hit the iteration limit without fully converging. I kept the lab's required configuration and did not add a scaler. The test accuracy is still good (0.9415), but the coefficients are probably not at the exact optimum. Scaling the features (e.g. StandardScaler) would likely fix the warning.
- RSS is high (~227 MB) because it measures the whole Python process, not just the model. The interpreter, NumPy, SciPy, scikit-learn and the other imported libraries take up almost all of that memory. The models themselves are tiny (1 KB and 284 KB), so the RSS delta during training and inference is only 0.0–0.1 MB, and the two models differ by only ~1.5 MB.
- Latency was measured on my own laptop (Windows 10): [write your CPU and RAM here], running a single Python process. The absolute numbers would be different on a phone or a microcontroller, so they should be read as relative comparisons.

### Deployment fit

| Target | LogisticRegression | RandomForest |
|---|---|---|
| Cloud | Fits. Latency 0.0524 ms ≪ 100 ms, size 1.03 KB ≪ 500 MB, and the ~227 MB process fits easily in ≥ 1 GB. | Fits. Latency 2.0155 ms ≪ 100 ms, size 284.07 KB ≪ 500 MB, and the ~228 MB process fits in ≥ 1 GB. |
| Edge | Fits. Latency 0.0524 ms ≤ 50 ms, size 1.03 KB ≤ 50 MB, and ~227 MB RSS is below even the 256 MB lower bound of the memory range. | Fits. Latency 2.0155 ms ≤ 50 ms, size 284.07 KB ≤ 50 MB, and ~228 MB RSS is below 256 MB. |
| Mobile | Fits on latency (0.0524 ms ≤ 20 ms) and size (1.03 KB ≤ 10 MB). Memory is tight: ~227 MB is only just inside the 64–256 MB range and would not fit on a low-end device. In practice the Python stack would be dropped and the model exported, which removes most of this overhead. | Same picture: latency (2.0155 ms ≤ 20 ms) and size (284.07 KB ≤ 10 MB) are fine, and memory is tight at ~228 MB out of 64–256 MB. The same caveat about exporting the model applies. |
| TinyML | Partly fits. Latency (0.0524 ms ≤ 10 ms) and size (1.03 KB ≤ 100 KB) pass. Measured memory fails: ~227 MB vs ≤ 256 KB, about 900× too much. That cost comes from the Python runtime, not the model, since a linear model with 30 coefficients is only a few hundred bytes. So it could run on TinyML only if re-implemented without Python (e.g. in C). | Does not fit. Size fails: 284.07 KB is about 2.8× over the 100 KB limit. Memory also fails (~228 MB vs ≤ 256 KB). Latency passes (2.0155 ms ≤ 10 ms) but only on my laptop CPU, and a microcontroller would be much slower. |

## 4. Conclusions
1. Both models are very cheap in absolute terms and meet the Cloud and Edge budgets with a large margin. Their accuracy is almost the same (0.9415 vs 0.9357). With a 171-sample test set, that gap is about one sample, so it is not a meaningful difference.
2. LogisticRegression is clearly the more efficient model: about 38× lower single-sample latency (0.0524 ms vs 2.0155 ms) and about 276× smaller (1055 bytes vs 290889 bytes), with no loss in accuracy. Training time is nearly the same (0.1551 s vs 0.1610 s), so it is not a deciding factor.
3. Process RSS is dominated by the Python and library overhead (~227 MB), not by the model, so it does not tell the models apart. This overhead is what makes Mobile memory tight and rules out TinyML for both models as measured. Only LogisticRegression could realistically reach TinyML, and only if it were re-implemented without the Python stack.