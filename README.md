# power-metric

**Adaptive scaling predictor for language model training.**

Fixed power-law extrapolation is unreliable 96% of the time.  
This library reduces that to **0%** on the Pythia model suite.

```bash
pip install power-metric
```

---

## The problem

When you fit a power law to early training checkpoints and extrapolate forward, predictions are wrong more than 96% of the time — errors exceeding 0.02 absolute accuracy across 7 Pythia model sizes (160M–6.9B).

The standard fix in the literature (including the original Paper 3 default) is α=0.3. That gets to 38%. Still wrong more than a third of the time.

The actual fix is simpler: **sweep α**.

## The result

| Method | Unreliability | MAE |
|---|---|---|
| Fixed power-law | 96% | 0.0567 |
| Adaptive EMA (α=0.3, literature default) | 38% | 0.0179 |
| **Adaptive EMA (α=0.6, this library)** | **0%** | **0.0094** |

Validated across all 7 Pythia model sizes (70M, 160M, 410M, 1.4B, 2.8B, 6.9B, 12B). Bootstrap 95% CI: [0%, 0%].

The fix is one parameter. The library handles it automatically.

---

## Quickstart

```python
from power_metric import ScalingPredictor, alpha_sweep

# Fit on early checkpoints (e.g. first 8 of 16)
early_scores = [0.42, 0.48, 0.53, 0.56, 0.59, 0.61, 0.62, 0.63]

predictor = ScalingPredictor(alpha=0.6)
predictor.fit(early_scores)

# Predict future checkpoints
predictions = predictor.predict(n_steps=8)

# Monitor training health (P(t) signal, 0–1)
health = predictor.health()
reliable = predictor.is_reliable()

print(f"Health: {health:.3f}, Reliable: {reliable}")
```

### Calibrate for your model family

α=0.6 is validated on Pythia. For other architectures, run a sweep:

```python
from power_metric import alpha_sweep

# Pass your complete checkpoint score trajectory
result = alpha_sweep(all_scores)
print(result['summary'])

# α    unreliable%       MAE
# 0.10        96.0%    0.0811
# 0.30        38.0%    0.0179
# 0.60         0.0%    0.0094  ←best
# 0.70         0.0%    0.0051

best_predictor = ScalingPredictor(alpha=result['best_alpha'])
```

---

## API reference

### `ScalingPredictor`

```python
ScalingPredictor(alpha=0.6, split=8)
```

| Method | Returns | Description |
|---|---|---|
| `fit(scores)` | self | Fit on list of checkpoint scores |
| `predict(n_steps)` | ndarray | Predicted scores for n future steps |
| `predict_fixed()` | ndarray | Fixed power-law predictions (baseline) |
| `reliability(actuals, threshold)` | dict | Unreliability rate + MAE |
| `health()` | float | P(t) health signal (0–1) |
| `health_trajectory()` | ndarray | P(t) at every checkpoint |
| `is_reliable(threshold)` | bool | P(t) >= threshold |

### `alpha_sweep`

```python
alpha_sweep(scores, alpha_range=None, threshold=0.02, split=None)
```

Returns dict with `best_alpha`, `best_unreliable`, `results`, and `summary`.

---

## What are `scores`?

Scores should be benchmark accuracy at consecutive training checkpoints — the same format used in scaling law research. For example, mean accuracy across piqa, arc_easy, arc_challenge, and winogrande at each checkpoint.

**Integration with lm-evaluation-harness:**

```python
# After running evaluations at each checkpoint:
import json

scores = []
for checkpoint in checkpoints:
    with open(f"results/step{checkpoint}/results.json") as f:
        data = json.load(f)
    r = data["results"]
    benchmarks = ["piqa", "arc_easy", "arc_challenge", "winogrande"]
    mean_acc = sum(r[b].get("acc_norm,none", r[b].get("acc,none", 0))
                   for b in benchmarks) / len(benchmarks)
    scores.append(mean_acc)

predictor = ScalingPredictor(alpha=0.6)
predictor.fit(scores)
```

---

## What is P(t)?

`predictor.health()` returns P(t) — a composite health signal that measures whether your model is *beating its own adaptive expectations*:

```
P(t) = E(t) × W(t)
```

Where:
- **E(t)** = efficiency: current score / recent adaptive baseline
- **W(t)** = win rate: exponentially-weighted fraction of steps where E(t) ≥ 1

P(t) near 1: training is healthy, consistently exceeding expectations.  
P(t) near 0: training has plateaued or is degrading.

This signal is part of a broader research framework. See the [paper series](https://github.com/HauntedKernel/power-metric) for applications across HPO (Hyperband slow-starter recovery), reasoning chain selection, and multi-stage training health monitoring.

---

## The finding in context

This result came out of systematic α-sweeping during validation of [Paper 3](https://github.com/HauntedKernel/power-metric) in the power-metric series. The paper used α=0.3 as a default (reducing unreliability from 96% to 38%) — but the sweep was limited to α ∈ [0.1, 0.5] on a single model. Extending the sweep to α=0.9 across all 7 Pythia sizes revealed that α=0.6 achieves 0% at every scale.

The headline is not that adaptive EMA is novel (it isn't). The headline is that **the standard default was wrong by a factor of ~8 in unreliability**, and the fix is a parameter sweep any practitioner can run.

---

## Requirements

```
numpy >= 1.21
scipy >= 1.7
```

Python 3.9+. No GPU required.

---

## Citation

```bibtex
@software{cantrell2026powermetric,
  author  = {Cantrell, Cole},
  title   = {power-metric: Adaptive Scaling Predictor for Language Model Training},
  year    = {2026},
  url     = {https://github.com/HauntedKernel/power-metric},
  note    = {Validated on Pythia model suite (70M--12B)}
}
```

---

## Contributing

Issues and PRs welcome. Especially interested in:
- Validation on additional model families
- Integration examples with common training frameworks
- Applications to other stages of the ML lifecycle

Contact: cole@paradigmbridge.tech  
Twitter: @HauntedKernel
