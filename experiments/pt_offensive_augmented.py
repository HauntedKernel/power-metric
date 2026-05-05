"""
Paper 3 — Offensive Augmentation with Smoothed Ratchet
========================================================

Reframing: Paper 3's existing "adaptive EMA" result (38% unreliability vs
96% for fixed power-law) was secretly offensive work — it acknowledged the
prior was wrong and let the prediction adapt. We're now making that
explicit and testing whether a *principled* offensive augmentation does
better than the implicit version.

The constraint Cole articulated:
  1. Sustained outperformance should eventually lift the benchmark
  2. Single-checkpoint outliers should NOT meaningfully move it
  3. Once moved, the benchmark should NOT snap back on normal checkpoints
  4. Symmetric (works in both directions)

The mathematical shape: a smoothed ratchet on the deviation signal,
applied as augmentation to the power-law prior.

  deviation(t)      = actual(t) - prior(t)          # noisy per-step
  smoothed_dev(t)   = (1-γ)·smoothed_dev(t-1) + γ·deviation(t)
  adjusted_prior(t) = prior(t) + smoothed_dev(t-1)  # use PRE-update for prediction

With γ small, single outliers don't move the correction much. Sustained
signal accumulates. Once accumulated, normal checkpoints don't snap it
back because each one only contributes γ-fraction of new correction.

Two framings tested:
  A (multiplicative): smoothed_ratio(t) tracks actual/prior, prior×ratio
  B (additive):       smoothed_dev(t) tracks actual-prior, prior+dev

Comparison bar:
  Fixed power-law:           96% unreliability (Paper 3 baseline, bad)
  Adaptive EMA (defensive):  38% unreliability (Paper 3 win, implicit offensive)
  Offensive A/B (this):      target — beat 38% with explicit offensive

Run from C:\\Users\\Carolina\\:
    python pt_offensive_augmented.py
"""

import os, json, re
import numpy as np
from scipy.optimize import curve_fit

BASE = './pythia-main/evals/pythia-v1'
BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge']
TARGET_MODELS = ['pythia-70m', 'pythia-160m', 'pythia-410m',
                 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b']
SPLIT = 8
UNRELIABILITY_THRESH = 0.10  # |pred-actual|/actual > 10% = unreliable


# ── Data loading ─────────────────────────────────────────────────────

def load_curve(model):
    folder = os.path.join(BASE, model, 'zero-shot')
    if not os.path.exists(folder):
        return None
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
    return np.array([step_scores[s] for s in steps])


def power_law(x, a, b, c):
    return a * np.power(x, b) + c


def fit_power_law_prior(scores, split=SPLIT):
    x_fit = np.arange(1, split + 1, dtype=float)
    x_full = np.arange(1, len(scores) + 1, dtype=float)
    try:
        popt, _ = curve_fit(
            power_law, x_fit, scores[:split],
            p0=[0.1, 0.3, 0.3], maxfev=10000,
            bounds=([-np.inf, -2, 0], [np.inf, 2, 1])
        )
        return power_law(x_full, *popt)
    except RuntimeError:
        return np.full(len(scores), scores[split - 1])


# ── Baselines ────────────────────────────────────────────────────────

def predict_fixed(scores, split=SPLIT):
    """Fixed power-law extrapolation — Paper 3's bad baseline (96%)."""
    prior = fit_power_law_prior(scores, split)
    return prior[split:], scores[split:]


def predict_adaptive_ema(scores, alpha=0.3, split=SPLIT):
    """Adaptive EMA — Paper 3's defensive variant (38%)."""
    er = scores[0]
    er_series = [er]
    for s in scores[1:]:
        er = (1 - alpha) * er + alpha * s
        er_series.append(er)
    return np.array(er_series[split - 1:-1]), scores[split:]


# ── Offensive variants ──────────────────────────────────────────────

def predict_offensive_additive(scores, gamma, split=SPLIT):
    """
    Framing B: additive smoothed-ratchet correction.

      deviation(t)    = actual(t) - prior(t)
      smoothed_dev(t) = (1-γ)·smoothed_dev(t-1) + γ·deviation(t)
      adjusted(t+1)   = prior(t+1) + smoothed_dev(t)   [pre-update; use last
                                                         smoothed before
                                                         observing actual(t+1)]

    The correction is a slow-moving accumulator on the deviation signal.
    """
    prior = fit_power_law_prior(scores, split)
    smoothed_dev = 0.0
    adjusted_predictions = np.zeros(len(scores) - split)

    # Initialize correction by walking through fitting region
    # (so we enter the test region with a calibrated correction)
    for t in range(split):
        dev = scores[t] - prior[t]
        smoothed_dev = (1 - gamma) * smoothed_dev + gamma * dev

    # Predict and update through test region.
    # Critical: prediction at t uses smoothed_dev FROM PREVIOUS step
    # (we don't get to peek at actual(t) before predicting it).
    for i, t in enumerate(range(split, len(scores))):
        # Predict using state from before observing actual(t)
        adjusted_predictions[i] = prior[t] + smoothed_dev
        # Now observe actual(t) and update correction
        dev = scores[t] - prior[t]
        smoothed_dev = (1 - gamma) * smoothed_dev + gamma * dev

    return adjusted_predictions, scores[split:]


def predict_offensive_multiplicative(scores, gamma, split=SPLIT):
    """
    Framing A: multiplicative smoothed-ratchet correction.

      ratio(t)         = actual(t) / prior(t)
      smoothed_ratio(t) = (1-γ)·smoothed_ratio(t-1) + γ·ratio(t)
      adjusted(t+1)    = prior(t+1) × smoothed_ratio(t)
    """
    prior = fit_power_law_prior(scores, split)
    smoothed_ratio = 1.0
    adjusted_predictions = np.zeros(len(scores) - split)

    # Calibrate through fitting region
    for t in range(split):
        ratio = scores[t] / max(prior[t], 1e-6)
        smoothed_ratio = (1 - gamma) * smoothed_ratio + gamma * ratio

    # Predict and update through test region
    for i, t in enumerate(range(split, len(scores))):
        adjusted_predictions[i] = prior[t] * smoothed_ratio
        ratio = scores[t] / max(prior[t], 1e-6)
        smoothed_ratio = (1 - gamma) * smoothed_ratio + gamma * ratio

    return adjusted_predictions, scores[split:]


# ── Reliability ─────────────────────────────────────────────────────

def reliability(predictions, actuals):
    """Fraction of predictions where |pred - actual| / actual > 10%."""
    err = np.abs(predictions - actuals) / np.maximum(actuals, 1e-6)
    return float(np.mean(err > UNRELIABILITY_THRESH))


def mae(predictions, actuals):
    return float(np.mean(np.abs(predictions - actuals)))


# ── Experiment ───────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("Paper 3 Offensive Augmentation — Explicit Smoothed-Ratchet Correction")
    print("=" * 90)
    print()
    print("Approach: take fixed power-law prior, add a smoothed-deviation correction")
    print("that lifts predictions when actuals consistently exceed prior, and lowers")
    print("them when actuals consistently fall short. Outliers are damped by smoothing;")
    print("sustained signal accumulates.")
    print()

    curves = {}
    for m in TARGET_MODELS:
        c = load_curve(m)
        if c is not None:
            curves[m] = c
    print(f"Loaded {len(curves)} configs, {len(next(iter(curves.values())))} ckpts each\n")

    # Baselines
    fixed_predictions = []; fixed_actuals = []
    for m, scores in curves.items():
        p, a = predict_fixed(scores)
        fixed_predictions.extend(p); fixed_actuals.extend(a)
    fixed_predictions = np.array(fixed_predictions)
    fixed_actuals = np.array(fixed_actuals)

    ema_predictions = []; ema_actuals = []
    for m, scores in curves.items():
        p, a = predict_adaptive_ema(scores, alpha=0.3)
        ema_predictions.extend(p); ema_actuals.extend(a)
    ema_predictions = np.array(ema_predictions)
    ema_actuals = np.array(ema_actuals)

    print(f"{'Method':<32} {'unreliability':>15} {'MAE':>10}")
    print("-" * 90)
    print(f"{'Fixed power-law (Paper 3 baseline)':<32} "
          f"{reliability(fixed_predictions, fixed_actuals)*100:>14.1f}% "
          f"{mae(fixed_predictions, fixed_actuals):>10.4f}")
    print(f"{'Adaptive EMA α=0.3 (Paper 3 win)':<32} "
          f"{reliability(ema_predictions, ema_actuals)*100:>14.1f}% "
          f"{mae(ema_predictions, ema_actuals):>10.4f}")
    print()

    # Sweep gamma for both offensive framings
    gamma_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    print("OFFENSIVE FRAMINGS — γ sweep")
    print("-" * 90)
    print(f"{'Framing':<22} {'γ':<6} {'unreliability':>15} {'MAE':>10} "
          f"{'Δ vs EMA':>12}")
    print("-" * 90)

    ema_unrel = reliability(ema_predictions, ema_actuals)

    for framing_name, fn in [
        ('Offensive_A_mult',  predict_offensive_multiplicative),
        ('Offensive_B_add',   predict_offensive_additive),
    ]:
        for gamma in gamma_grid:
            preds = []; acts = []
            for m, scores in curves.items():
                p, a = fn(scores, gamma=gamma)
                preds.extend(p); acts.extend(a)
            preds = np.array(preds); acts = np.array(acts)
            unrel = reliability(preds, acts)
            err = mae(preds, acts)
            delta = (unrel - ema_unrel) * 100
            print(f"{framing_name:<22} {gamma:<6.2f} {unrel*100:>14.1f}% "
                  f"{err:>10.4f} {delta:>+11.1f}%")
        print()

    # Per-model breakdown at best gamma for additive
    # (find best on aggregate)
    best_gamma_a = None; best_unrel_a = float('inf')
    best_gamma_b = None; best_unrel_b = float('inf')
    for gamma in gamma_grid:
        preds_a, acts_a = [], []
        preds_b, acts_b = [], []
        for m, scores in curves.items():
            p, a = predict_offensive_multiplicative(scores, gamma=gamma)
            preds_a.extend(p); acts_a.extend(a)
            p, a = predict_offensive_additive(scores, gamma=gamma)
            preds_b.extend(p); acts_b.extend(a)
        u_a = reliability(np.array(preds_a), np.array(acts_a))
        u_b = reliability(np.array(preds_b), np.array(acts_b))
        if u_a < best_unrel_a: best_unrel_a = u_a; best_gamma_a = gamma
        if u_b < best_unrel_b: best_unrel_b = u_b; best_gamma_b = gamma

    print("=" * 90)
    print(f"Per-model breakdown — Offensive A (γ={best_gamma_a}) vs "
          f"Offensive B (γ={best_gamma_b}) vs Adaptive EMA")
    print("=" * 90)
    print(f"\n{'Model':<14} {'Fixed%':>9} {'EMA%':>8} "
          f"{f'OffA(γ={best_gamma_a})%':>14} {f'OffB(γ={best_gamma_b})%':>14}")
    print("-" * 70)
    for m, scores in curves.items():
        pf, af = predict_fixed(scores)
        pe, ae = predict_adaptive_ema(scores, alpha=0.3)
        pa, _ = predict_offensive_multiplicative(scores, gamma=best_gamma_a)
        pb, _ = predict_offensive_additive(scores, gamma=best_gamma_b)
        print(f"{m:<14} "
              f"{reliability(pf, af)*100:>8.1f}% "
              f"{reliability(pe, ae)*100:>7.1f}% "
              f"{reliability(pa, af)*100:>13.1f}% "
              f"{reliability(pb, af)*100:>13.1f}%")

    # Show prediction trajectories for one config to visualize what augmentation does
    print("\n" + "=" * 90)
    print(f"Trajectory inspection — pythia-12b (γ=0.10 for both)")
    print("=" * 90)
    if 'pythia-12b' in curves:
        scores = curves['pythia-12b']
        prior = fit_power_law_prior(scores)
        pa, _ = predict_offensive_multiplicative(scores, gamma=0.10)
        pb, _ = predict_offensive_additive(scores, gamma=0.10)
        pe, _ = predict_adaptive_ema(scores, alpha=0.3)

        print(f"\n{'Ckpt':>4} {'Actual':>9} {'Fixed':>9} {'Δfix%':>7} "
              f"{'EMA':>9} {'Δema%':>7} {'OffA':>9} {'ΔoffA%':>7} "
              f"{'OffB':>9} {'ΔoffB%':>7}")
        print("-" * 90)
        for i in range(len(scores) - SPLIT):
            t = SPLIT + i
            actual = scores[t]
            print(f"{t+1:>4} {actual:>9.4f} "
                  f"{prior[t]:>9.4f} {(prior[t]-actual)/actual*100:>+6.1f}% "
                  f"{pe[i]:>9.4f} {(pe[i]-actual)/actual*100:>+6.1f}% "
                  f"{pa[i]:>9.4f} {(pa[i]-actual)/actual*100:>+6.1f}% "
                  f"{pb[i]:>9.4f} {(pb[i]-actual)/actual*100:>+6.1f}%")

    print("\n" + "=" * 90)
    print("INTERPRETATION GUIDE")
    print("=" * 90)
    print("""
  Read the gamma sweep table:

  STRONG positive: at some γ, offensive A or B beats EMA's 38% by a meaningful
    margin (say below 30%). → explicit offensive augmentation outperforms
    implicit version. Real win for the framework story.

  COMPARABLE: best offensive variants land near 38%. → augmentation matches
    EMA's implicit work but doesn't surpass it. Both are doing the same
    thing under different framings. Catalog gets two equivalent variants.

  WORSE: offensive variants stay above 38% across all γ. → augmentation
    introduces noise or lag that hurts more than the prior correction
    helps. Defensive EMA remains the right tool for Paper 3.

  Look at the trajectory inspection table to see what offensive is doing
  visibly: is it tracking actual closer than fixed but smoother than EMA?
  Or is it lagging, oscillating, snapping back?

  For γ choice: small γ = slow ratchet, only sustained signal moves it.
  Large γ = fast adjustment, vulnerable to noise. The sweet spot will tell
  us about the noise-vs-responsiveness tradeoff for this kind of data.
""")


if __name__ == '__main__':
    main()
