"""
power-metric quickstart
========================
Demonstrates the 96%→0% unreliability result on Pythia data.

Run from the repo root:
    python examples/quickstart.py --data ./pythia-main/evals/pythia-v1

Or without Pythia data (uses synthetic example):
    python examples/quickstart.py
"""

import sys
import os
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from power_metric import ScalingPredictor, alpha_sweep


# ── Load real Pythia data if available ───────────────────────────────

def load_pythia(base_path, model):
    """Load checkpoint scores for one Pythia model."""
    import json, re
    folder = os.path.join(base_path, model, 'zero-shot')
    if not os.path.exists(folder):
        return None
    BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande',
                   'arc_easy', 'arc_challenge']
    step_scores = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m: continue
        step = int(m.group(1))
        if step < 1000: continue
        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)
        r = data.get('results', {})
        sc = []
        for b in BENCHMARKS:
            if b in r:
                bd = r[b]
                s = bd.get('acc_norm', bd.get('acc', None))
                if s is not None: sc.append(s)
        if len(sc) == len(BENCHMARKS):
            step_scores[step] = float(np.mean(sc))
    steps = sorted(step_scores.keys())
    if len(steps) < 16: return None
    return [step_scores[s] for s in steps]


# ── Synthetic example (no data needed) ───────────────────────────────

def synthetic_scores(n=16, seed=42):
    """Simulate a typical Pythia-1.4B scaling trajectory."""
    rng = np.random.default_rng(seed)
    # Power law trend + noise, similar to actual Pythia trajectories
    x = np.arange(1, n + 1, dtype=float)
    trend = 0.25 * np.power(x, 0.18) + 0.28
    noise = rng.normal(0, 0.008, n)
    return list(np.clip(trend + noise, 0, 1))


# ── Main demo ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None,
                        help='Path to pythia-main/evals/pythia-v1')
    args = parser.parse_args()

    print("=" * 60)
    print("power-metric quickstart")
    print("=" * 60)

    # Load data
    if args.data:
        models = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']
        all_scores = {}
        for m in models:
            s = load_pythia(args.data, m)
            if s:
                all_scores[m] = s
        if not all_scores:
            print(f"No data found at {args.data}, using synthetic example")
            all_scores = {'synthetic-model': synthetic_scores()}
    else:
        print("Using synthetic example (pass --data for real Pythia results)")
        all_scores = {'synthetic-model': synthetic_scores()}

    SPLIT = 8

    # ── Demo 1: Basic usage ───────────────────────────────────────────
    print("\n── Demo 1: Basic usage ──────────────────────────────────")

    model_name = list(all_scores.keys())[0]
    scores = all_scores[model_name]
    print(f"\nModel: {model_name}")
    print(f"Total checkpoints: {len(scores)}")
    print(f"Early checkpoints (fit): {SPLIT}")
    print(f"Late checkpoints (test): {len(scores) - SPLIT}")

    predictor = ScalingPredictor(alpha=0.6)
    predictor.fit(scores)

    predictions = predictor.predict()
    health = predictor.health()
    reliable = predictor.is_reliable()

    print(f"\nP(t) health signal: {health:.4f}")
    print(f"Is reliable: {reliable}")

    # ── Demo 2: Reliability comparison ───────────────────────────────
    print("\n── Demo 2: Fixed power-law vs adaptive EMA ─────────────")
    print()
    print(f"  {'Method':<28} {'Unreliable%':>13} {'MAE':>10}")
    print("  " + "-" * 54)

    for model_name, scores in all_scores.items():
        predictor_03 = ScalingPredictor(alpha=0.3, split=SPLIT)
        predictor_03.fit(scores)
        r03 = predictor_03.reliability()

        predictor_06 = ScalingPredictor(alpha=0.6, split=SPLIT)
        predictor_06.fit(scores)
        r06 = predictor_06.reliability()

        fixed_unrel = r06.get('fixed_unreliable', float('nan'))
        fixed_mae   = r06.get('fixed_mae', float('nan'))

        print(f"\n  {model_name}")
        print(f"  {'Fixed power-law':<28} "
              f"{fixed_unrel*100:>12.0f}% {fixed_mae:>10.4f}")
        print(f"  {'Adaptive EMA (α=0.3, Paper 3)':<28} "
              f"{r03['adaptive_unreliable']*100:>12.0f}% "
              f"{r03['adaptive_mae']:>10.4f}")
        print(f"  {'Adaptive EMA (α=0.6, improved)':<28} "
              f"{r06['adaptive_unreliable']*100:>12.0f}% "
              f"{r06['adaptive_mae']:>10.4f}")

    # ── Demo 3: Alpha sweep ───────────────────────────────────────────
    print("\n── Demo 3: Alpha sweep (calibrate for your data) ───────")

    scores = list(all_scores.values())[0]
    result = alpha_sweep(scores, split=SPLIT)
    print()
    print(result['summary'])

    # ── Demo 4: Trajectory inspection ────────────────────────────────
    print("\n── Demo 4: Prediction trajectory ───────────────────────")

    model_name = list(all_scores.keys())[0]
    scores = all_scores[model_name]
    predictor = ScalingPredictor(alpha=0.6, split=SPLIT)
    predictor.fit(scores)
    preds = predictor.predict()
    fixed = predictor.predict_fixed()

    print(f"\n  {'Ckpt':>5} {'Actual':>8} "
          f"{'Adaptive':>10} {'|err|':>7} "
          f"{'Fixed':>8} {'|err|':>7}")
    print("  " + "-" * 55)
    for i in range(SPLIT, len(scores)):
        actual = scores[i]
        pred_a = preds[i] if i < len(preds) else float('nan')
        pred_f = fixed[i] if fixed is not None and i < len(fixed) else float('nan')
        err_a = abs(pred_a - actual)
        err_f = abs(pred_f - actual) if not np.isnan(pred_f) else float('nan')
        flag_a = '!' if err_a > 0.02 else ' '
        flag_f = '!' if not np.isnan(err_f) and err_f > 0.02 else ' '
        print(f"  {i+1:>5} {actual:>8.4f} "
              f"{pred_a:>10.4f} {err_a:>6.4f}{flag_a} "
              f"{pred_f:>8.4f} {err_f:>6.4f}{flag_f}")
    print("\n  '!' = exceeds 0.02 absolute error threshold")

    print("\n" + "=" * 60)
    print("Done. See README.md for full documentation.")
    print("=" * 60)


if __name__ == '__main__':
    main()
