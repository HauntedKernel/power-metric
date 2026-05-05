"""
Paper 3 Offensive — Robustness Checks on the 25% Result
=========================================================

Three checks before declaring offensive variant a real win:

  1. Bootstrap 95% CI on the unreliability rate. With only 24 post-split
     observations across 3 configs, the 38% vs 25% difference might be
     inside noise. We need to know.

  2. EMA at α=0.5 (matching offensive's responsiveness). If fast-EMA also
     beats slow-EMA, then offensive's win is "fast tracking" not "offensive
     structure." Doesn't change Cole's stated criterion (any win counts),
     but tells us what's actually doing the work.

  3. Generalization test. Run on full 7-config Pythia set, not just Paper 3's
     selected 3. If offensive holds, robust. If only wins on 3-config subset,
     suspicious.

Run from C:\\Users\\Carolina\\:
    python pt_offensive_checks.py
"""

import os, json, re
import numpy as np
from scipy.optimize import curve_fit

BASE = './pythia-main/evals/pythia-v1'
BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge']

PAPER3_MODELS = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']
ALL_MODELS = ['pythia-70m', 'pythia-160m', 'pythia-410m',
              'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b']
SPLIT = 8
UNRELIABILITY_THRESH = 0.02

N_BOOTSTRAP = 10000
RNG_SEED = 42


# ── Loading ──────────────────────────────────────────────────────────

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


def fit_power_law(scores, split=SPLIT):
    x_fit = np.arange(1, split + 1, dtype=float)
    x_full = np.arange(1, len(scores) + 1, dtype=float)
    try:
        popt, _ = curve_fit(power_law, x_fit, scores[:split],
                             p0=[0.1, 0.3, 0.3], maxfev=10000,
                             bounds=([-np.inf, -2, 0], [np.inf, 2, 1]))
        return power_law(x_full, *popt)
    except RuntimeError:
        return np.full(len(scores), scores[split - 1])


# ── Predictors ───────────────────────────────────────────────────────

def predict_fixed(scores, split=SPLIT):
    return fit_power_law(scores, split)[split:], scores[split:]


def predict_ema(scores, alpha, split=SPLIT):
    er = scores[0]
    er_series = [er]
    for s in scores[1:]:
        er = (1 - alpha) * er + alpha * s
        er_series.append(er)
    return np.array(er_series[split - 1:-1]), scores[split:]


def predict_off_add(scores, gamma, split=SPLIT):
    prior = fit_power_law(scores, split)
    sd = 0.0
    for t in range(split):
        sd = (1 - gamma) * sd + gamma * (scores[t] - prior[t])
    out = np.zeros(len(scores) - split)
    for i, t in enumerate(range(split, len(scores))):
        out[i] = prior[t] + sd
        sd = (1 - gamma) * sd + gamma * (scores[t] - prior[t])
    return out, scores[split:]


def predict_off_mult(scores, gamma, split=SPLIT):
    prior = fit_power_law(scores, split)
    sr = 1.0
    for t in range(split):
        sr = (1 - gamma) * sr + gamma * (scores[t] / max(prior[t], 1e-6))
    out = np.zeros(len(scores) - split)
    for i, t in enumerate(range(split, len(scores))):
        out[i] = prior[t] * sr
        sr = (1 - gamma) * sr + gamma * (scores[t] / max(prior[t], 1e-6))
    return out, scores[split:]


# ── Metrics ──────────────────────────────────────────────────────────

def errors(preds, actuals):
    return np.abs(np.asarray(preds) - np.asarray(actuals))


def unreliability_rate(preds, actuals, threshold=UNRELIABILITY_THRESH):
    return float(np.mean(errors(preds, actuals) > threshold))


def collect_errors(predict_fn, models, **kwargs):
    """Run predict_fn over all models, return concatenated absolute errors."""
    all_err = []
    for m in models:
        scores = load_curve(m)
        if scores is None:
            continue
        preds, acts = predict_fn(scores, **kwargs)
        all_err.extend(errors(preds, acts).tolist())
    return np.array(all_err)


# ── Check 1: Bootstrap CI ────────────────────────────────────────────

def bootstrap_ci(errs, threshold=UNRELIABILITY_THRESH, n_boot=N_BOOTSTRAP,
                  rng_seed=RNG_SEED):
    """Bootstrap 95% CI on unreliability rate."""
    rng = np.random.default_rng(rng_seed)
    n = len(errs)
    rates = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(errs, size=n, replace=True)
        rates[i] = np.mean(sample > threshold)
    return float(np.mean(rates)), float(np.percentile(rates, 2.5)), \
           float(np.percentile(rates, 97.5))


def bootstrap_diff_ci(errs_a, errs_b, threshold=UNRELIABILITY_THRESH,
                       n_boot=N_BOOTSTRAP, rng_seed=RNG_SEED):
    """
    Bootstrap CI on the DIFFERENCE in unreliability rates (a - b).
    Pairs samples by index since both methods are run on the same checkpoints.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(errs_a)
    assert len(errs_b) == n, "paired bootstrap requires same length"
    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rate_a = np.mean(errs_a[idx] > threshold)
        rate_b = np.mean(errs_b[idx] > threshold)
        diffs[i] = rate_a - rate_b
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), \
           float(np.percentile(diffs, 97.5))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("Paper 3 Offensive — Robustness Checks")
    print("=" * 90)

    # ─── Check 1: Bootstrap CI on Paper 3 subset (3 configs) ─────────
    print()
    print("=" * 90)
    print("CHECK 1: Bootstrap 95% CI — Paper 3 subset (3 configs, 24 obs)")
    print("=" * 90)
    print()

    err_fixed = collect_errors(predict_fixed, PAPER3_MODELS)
    err_ema   = collect_errors(predict_ema, PAPER3_MODELS, alpha=0.3)
    err_offA  = collect_errors(predict_off_mult, PAPER3_MODELS, gamma=0.5)
    err_offB  = collect_errors(predict_off_add, PAPER3_MODELS, gamma=0.5)

    print(f"{'Method':<22} {'rate':>8} {'95% CI':>20} {'n_obs':>7}")
    print("-" * 60)
    for label, errs in [
        ('Fixed power-law', err_fixed),
        ('EMA α=0.3 (Paper 3)', err_ema),
        ('Offensive A γ=0.5', err_offA),
        ('Offensive B γ=0.5', err_offB),
    ]:
        rate, lo, hi = bootstrap_ci(errs)
        print(f"{label:<22} {rate*100:>7.0f}% "
              f"[{lo*100:>5.1f}, {hi*100:>5.1f}] {len(errs):>7}")

    # Paired bootstrap on the difference
    print()
    print("Paired bootstrap on differences (Offensive - EMA):")
    print()
    print(f"  {'Comparison':<32} {'mean Δ':>9} {'95% CI':>20} "
          f"{'p(Δ<0)':>10}")
    print("  " + "-" * 70)
    for label, errs in [
        ('OffA γ=0.5  vs  EMA α=0.3', err_offA),
        ('OffB γ=0.5  vs  EMA α=0.3', err_offB),
    ]:
        mean_diff, lo, hi = bootstrap_diff_ci(errs, err_ema)
        # p(off better) = P(diff < 0) where diff = off_rate - ema_rate
        rng = np.random.default_rng(RNG_SEED)
        n = len(errs)
        diffs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, size=n, replace=True)
            d = np.mean(errs[idx] > UNRELIABILITY_THRESH) - \
                np.mean(err_ema[idx] > UNRELIABILITY_THRESH)
            diffs.append(d)
        p_off_better = np.mean(np.array(diffs) < 0)
        sig = "✓" if (lo > 0 or hi < 0) else "✗"
        print(f"  {label:<32} {mean_diff*100:>+8.1f}% "
              f"[{lo*100:>+5.1f}, {hi*100:>+5.1f}] "
              f"{p_off_better:>9.3f} {sig}")

    print()
    print("  '✓' = 95% CI excludes 0 (statistically significant)")
    print("  '✗' = CI includes 0 (could be noise)")
    print("  p(Δ<0) = bootstrap probability that offensive beats EMA")

    # ─── Check 2: EMA with α=0.5 ────────────────────────────────────
    print()
    print("=" * 90)
    print("CHECK 2: Is the win 'offensive structure' or just 'fast tracking'?")
    print("=" * 90)
    print()
    print("Run EMA at multiple α values to see if fast-α alone matches")
    print("offensive's performance. If yes, offensive's win is responsiveness.")
    print("Either way, this doesn't undermine the result — just tells us")
    print("what mechanism is doing the work.")
    print()

    print(f"  {'Method':<22} {'rate':>8} {'95% CI':>20} {'MAE':>10}")
    print("  " + "-" * 70)
    for alpha in [0.1, 0.3, 0.5, 0.7]:
        errs = collect_errors(predict_ema, PAPER3_MODELS, alpha=alpha)
        rate, lo, hi = bootstrap_ci(errs)
        mae = float(np.mean(errs))
        print(f"  EMA α={alpha:<16.1f} {rate*100:>7.0f}% "
              f"[{lo*100:>5.1f}, {hi*100:>5.1f}] {mae:>10.4f}")
    print()
    rate_a, lo_a, hi_a = bootstrap_ci(err_offA)
    mae_a = float(np.mean(err_offA))
    print(f"  {'Offensive A γ=0.5':<22} {rate_a*100:>7.0f}% "
          f"[{lo_a*100:>5.1f}, {hi_a*100:>5.1f}] {mae_a:>10.4f}")
    rate_b, lo_b, hi_b = bootstrap_ci(err_offB)
    mae_b = float(np.mean(err_offB))
    print(f"  {'Offensive B γ=0.5':<22} {rate_b*100:>7.0f}% "
          f"[{lo_b*100:>5.1f}, {hi_b*100:>5.1f}] {mae_b:>10.4f}")

    # ─── Check 3: Generalization to all 7 configs ───────────────────
    print()
    print("=" * 90)
    print("CHECK 3: Does the win generalize? Full 7-config Pythia set")
    print("=" * 90)
    print()

    err_fixed_all = collect_errors(predict_fixed, ALL_MODELS)
    err_ema_all   = collect_errors(predict_ema, ALL_MODELS, alpha=0.3)
    err_offA_all  = collect_errors(predict_off_mult, ALL_MODELS, gamma=0.5)
    err_offB_all  = collect_errors(predict_off_add, ALL_MODELS, gamma=0.5)

    print(f"  {'Method':<22} {'rate':>8} {'95% CI':>20} {'n_obs':>7}")
    print("  " + "-" * 70)
    for label, errs in [
        ('Fixed power-law', err_fixed_all),
        ('EMA α=0.3', err_ema_all),
        ('Offensive A γ=0.5', err_offA_all),
        ('Offensive B γ=0.5', err_offB_all),
    ]:
        rate, lo, hi = bootstrap_ci(errs)
        print(f"  {label:<22} {rate*100:>7.0f}% "
              f"[{lo*100:>5.1f}, {hi*100:>5.1f}] {len(errs):>7}")

    print()
    print("  Paired bootstrap (Offensive - EMA) on full 7-config set:")
    print()
    for label, errs in [
        ('OffA γ=0.5  vs  EMA α=0.3', err_offA_all),
        ('OffB γ=0.5  vs  EMA α=0.3', err_offB_all),
    ]:
        mean_diff, lo, hi = bootstrap_diff_ci(errs, err_ema_all)
        sig = "✓" if (lo > 0 or hi < 0) else "✗"
        print(f"    {label:<32} {mean_diff*100:>+8.1f}% "
              f"[{lo*100:>+5.1f}, {hi*100:>+5.1f}] {sig}")

    # ─── Per-config breakdown on all 7 ──────────────────────────────
    print()
    print("  Per-config breakdown (all 7):")
    print()
    print(f"  {'Model':<14} {'Fixed%':>8} {'EMA%':>7} {'OffA%':>7} {'OffB%':>7}")
    print("  " + "-" * 50)
    for m in ALL_MODELS:
        scores = load_curve(m)
        if scores is None:
            continue
        pf, af = predict_fixed(scores)
        pe, _ = predict_ema(scores, alpha=0.3)
        pa, _ = predict_off_mult(scores, gamma=0.5)
        pb, _ = predict_off_add(scores, gamma=0.5)
        print(f"  {m:<14} "
              f"{unreliability_rate(pf, af)*100:>7.0f}% "
              f"{unreliability_rate(pe, af)*100:>6.0f}% "
              f"{unreliability_rate(pa, af)*100:>6.0f}% "
              f"{unreliability_rate(pb, af)*100:>6.0f}%")

    # ─── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("SUMMARY GUIDE")
    print("=" * 90)
    print("""
  CHECK 1 (paired bootstrap on Paper 3 subset):
    ✓ on both → offensive beats EMA significantly on the 3-config set
    ✗ on either → 25% vs 38% might be noise; underwhelming claim

  CHECK 2 (EMA at α=0.5):
    If EMA α=0.5 also achieves ~25%, the win is "fast tracking" not "offensive
    structure." That's still valid per Cole's stated criterion (any win
    stacks), but the framing should be honest about mechanism.
    If EMA α=0.5 stays at ~38% or worse, offensive variants are doing
    something EMA can't replicate just by tuning α.

  CHECK 3 (full 7-config generalization):
    ✓ on full set → robust win, generalizes beyond Paper 3's subset
    ✗ on full set but ✓ on subset → result is fragile to model selection
    Equal performance on both → consistent claim, just not a subset effect

  If all three checks support the result, we have a defensible narrow claim:
  "Smoothed-ratchet correction on power-law prior reduces unreliability vs
  adaptive EMA on Pythia scaling reliability."
""")


if __name__ == '__main__':
    main()
