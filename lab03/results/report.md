# Lab 3: Architectures and their costs

## 1. Goal

TODO: write 2-3 sentences in your own words. What did you compare (MLP, CNN, attention model on Fashion-MNIST) and which costs did you measure (parameters, FLOPs, latency, accuracy)?

## 2. Method

- Environment: Python 3.11.8 on Windows 10, Intel Core i5-1135G7 CPU, no GPU used. Package versions: `results/versions.txt`.
- Data: Fashion-MNIST, ToTensor + Normalize((0.2860,), (0.3530,)). Official train set split into 55k train and 5k validation (seed 42), official 10k test set. Batch size 128.
- Models: MLP, CNN and patch-based attention model exactly as specified in the lab manual. The attention model has no positional embedding, because the manual does not list one.
- Parameters: `torchinfo.summary` (full summaries in `results/torchinfo_*.txt`).
- FLOPs: one layer of each type calculated by hand (see the photo below), compared with `thop.profile`.
- Latency: CPU, eval mode, no gradients, 1 warm-up run, then median of 100 runs for batch 1 and median of 20 runs for batch 128.
- Training: Adam, learning rate 1e-3, batch 128, 5 epochs, seed 42. Test accuracy is measured once, after the last epoch.
- Code is in `src/`: `common.py`, `cost.py`, `train.py`, `plots.py`, `print_versions.py`.

## 3. Results

### 3.1 Parameters, accuracy and latency

| Model | Parameters | Test accuracy | Latency batch 1 (ms) | Latency batch 128 (ms) |
|---|---|---|---|---|
| MLP | 235,146 | 0.8755 | 0.0947 | 0.6378 |
| CNN | 20,490 | 0.8928 | 0.2684 | 13.0767 |
| Attention | 70,794 | 0.8435 | 0.6958 | 12.9748 |

Note: the manual targets 100k-150k parameters, but the specified architectures give 235k, 20k and 71k. I kept the architectures exactly as written in the manual.

### 3.2 Hand-calculated vs measured FLOPs

thop reports MACs (multiply-accumulate), so the thop column below is MACs x 2 to compare it with FLOPs.

| Layer | Hand-calculated FLOPs | thop MACs | thop MACs x 2 |
|---|---|---|---|
| MLP Linear 784 to 256 | TODO | 200,704 | 401,408 |
| CNN Conv2d 16 to 32 (14x14) | TODO | 903,168 | 1,806,336 |
| Attention QKV projection (seq 16) | TODO | 196,608 | 393,216 |
| Attention scores (seq 16, d_model 64) | TODO | not counted by thop | not counted by thop |

Whole models according to thop (MACs): MLP 234,752, CNN 1,031,744, Attention 591,488.

TODO: explain the gap between your hand numbers and thop in your own words.

Hand calculations (photo):

![Hand calculations](results/hand_calculations.jpg)

### 3.3 Trade-off plots

![Accuracy vs parameters](results/accuracy_vs_params.png)

![Accuracy vs latency](results/accuracy_vs_latency.png)

TODO: explain in your own words which architecture is most cost-effective, using the access patterns from Reddi Ch. 4.6-4.7.

## 4. Conclusions

1. TODO
2. TODO
3. TODO
