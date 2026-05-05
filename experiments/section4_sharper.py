"""
Section 4 v2 — Scaling Reliability with Sharper Tests
=====================================================
Adds to our earlier Pythia verification:
  1. Test against multiple fixed-PL fitting strategies (not just naive
     power law on first half) — addresses Lourie's critique that
     adaptive baseline = exponential smoothing
  2. Test the alpha-by-scale claim: does optimal α systematically
     scale with model size? Or is it noise?
  3. Test the per-checkpoint detection: at which exact checkpoint
     does fixed-PL diverge from actual, and does adaptive catch it?
  4. Compare adaptive E[R] vs Holt-Winters / ETS as time-series
     baselines (Lourie's specific suggestion)

Run:
    python section4_sharper.py
"""

import os, json, re
import numpy as np
from scipy.optimize import curve_fit

ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3
RNG_SEED = 42
N_BOOTSTRAP = 2000
THRESHOLD = 0.02

BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']
MODELS = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']
BASE = './pythia-main/evals/pythia-v1'


def load_composite(model):
    folder = os.path.join(BASE, model, 'zero-shot')
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
                if s is not None:
                    sc.append(s)
        if len(sc) == len(BENCHMARKS):
            step_scores[step] = float(np.mean(sc))
    steps = sorted(step_scores.keys())
    return steps, np.array([step_scores[s] for s in steps])


def power_law(x, a, b, c):
    return a * np.power(x, b) + c


def adaptive_baseline(scores, alpha):
    """Pre-update E[R](t) — the adaptive predictor."""
    er = None
    out = []
    for s in scores:
        prev = er if er is not None else s
        out.append(prev)
        er = s if er is None else (1-alpha)*er + alpha*s
    return np.array(out)


def holt_winters_predict(scores, n_fit, alpha=0.3, beta=0.1):
    """
    Simple Holt's linear trend method as an alternative to power-law fixed
    extrapolation. Lourie's point: adaptive E[R] is essentially exp.
    smoothing — we should compare against the time-series literature.
    """
    fit = scores[:n_fit]
    # Initialize
    level = fit[0]
    trend = fit[1] - fit[0] if len(fit) > 1 else 0
    for s in fit[1:]:
        new_level = alpha*s + (1-alpha)*(level+trend)
        new_trend = beta*(new_level - level) + (1-beta)*trend
        level, trend = new_level, new_trend
    # Forecast
    n_pred = len(scores) - n_fit
    return np.array([level + (h+1)*trend for h in range(n_pred)])


def ses_predict(scores, n_fit, alpha=0.3):
    """Simple exponential smoothing — no trend. Forecast is constant."""
    fit = scores[:n_fit]
    level = fit[0]
    for s in fit[1:]:
        level = alpha*s + (1-alpha)*level
    n_pred = len(scores) - n_fit
    return np.full(n_pred, level)


def main():
    print("="*78)
    print("Section 4 v2 — Sharper Scaling Reliability Tests")
    print("="*78)

    rng = np.random.default_rng(RNG_SEED)

    # 1. Compare adaptive vs Holt-Winters vs SES vs fixed PL
    print(f"\n{'─'*78}")
    print("[1] Predictor Comparison (8/8 split, threshold=0.02)")
    print(f"{'─'*78}")
    for model in MODELS:
        steps, scores = load_composite(model)
        n = len(scores)
        n_fit = n // 2
        actual = scores[n_fit:]
        x = np.arange(1, n+1)

        # Fixed PL
        try:
            popt, _ = curve_fit(power_law, x[:n_fit], scores[:n_fit],
                                p0=[1.0, -0.1, 0.5], maxfev=20000)
            fixed_pred = power_law(x[n_fit:], *popt)
        except Exception:
            fixed_pred = np.full(n-n_fit, scores[:n_fit].mean())

        # Adaptive at α=0.3 and α=0.5
        adapt_03 = adaptive_baseline(scores, 0.3)[n_fit:]
        adapt_05 = adaptive_baseline(scores, 0.5)[n_fit:]

        # Holt-Winters and SES
        hw_pred = holt_winters_predict(scores, n_fit)
        ses_pred = ses_predict(scores, n_fit)

        print(f"\n  {model}  (n={n}, fit={n_fit}, predict={n-n_fit})")
        print(f"  {'Predictor':<22} {'MAE':>8} {'Unrel':>8} {'95% CI':<18}")
        print(f"  {'-'*60}")
        for name, pred in [('Fixed power law', fixed_pred),
                           ('Adaptive E[R] α=0.3', adapt_03),
                           ('Adaptive E[R] α=0.5', adapt_05),
                           ('Holt-Winters', hw_pred),
                           ('Simple exp. smooth.', ses_pred)]:
            err = np.abs(pred - actual)
            mae = err.mean()
            unrel = (err > THRESHOLD).mean()
            boot = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, len(actual), len(actual))
                boot.append((err[idx] > THRESHOLD).mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {name:<22} {mae:>8.4f} {unrel*100:>7.0f}% "
                  f"[{lo*100:>3.0f}, {hi*100:>3.0f}]%")

    # 2. Alpha-by-scale: optimal alpha vs model size
    print(f"\n{'─'*78}")
    print("[2] Optimal α by Scale  (does it systematically increase with size?)")
    print(f"{'─'*78}")
    print(f"\n  Lourie's question: is α a tunable artifact, or does it scale meaningfully?")
    print(f"\n  {'Model':<14} {'Optimal α':>10} {'Min unrel%':>11} {'Min MAE':>9}")
    for model in MODELS:
        steps, scores = load_composite(model)
        n = len(scores)
        n_fit = n // 2
        actual = scores[n_fit:]
        best_a = None; best_unrel = 1.1; best_mae = 1.0
        for a in np.arange(0.05, 0.96, 0.05):
            pred = adaptive_baseline(scores, a)[n_fit:]
            err = np.abs(pred - actual)
            unrel = (err > THRESHOLD).mean()
            if unrel < best_unrel or (unrel == best_unrel and err.mean() < best_mae):
                best_unrel = unrel; best_a = a; best_mae = err.mean()
        print(f"  {model:<14} {best_a:>10.2f} {best_unrel*100:>10.0f}% {best_mae:>9.4f}")

    # 3. Per-checkpoint divergence — when does fixed PL go wrong?
    print(f"\n{'─'*78}")
    print("[3] Per-Checkpoint Errors (1.4B as example)")
    print(f"{'─'*78}")
    steps, scores = load_composite('pythia-1.4b')
    n = len(scores); n_fit = n // 2
    actual = scores[n_fit:]
    x = np.arange(1, n+1)
    popt, _ = curve_fit(power_law, x[:n_fit], scores[:n_fit],
                        p0=[1.0, -0.1, 0.5], maxfev=20000)
    fixed_pred = power_law(x[n_fit:], *popt)
    adapt_pred = adaptive_baseline(scores, 0.3)[n_fit:]

    print(f"\n  {'Ckpt':<5} {'Actual':>8} {'Fixed PL':>9} {'|err|':>7} "
          f"{'Adaptive':>9} {'|err|':>7}")
    for i in range(len(actual)):
        fe = abs(fixed_pred[i] - actual[i])
        ae = abs(adapt_pred[i] - actual[i])
        fmark = ' *' if fe > THRESHOLD else '  '
        amark = ' *' if ae > THRESHOLD else '  '
        print(f"  {n_fit+i:<5} {actual[i]:>8.4f} {fixed_pred[i]:>9.4f} "
              f"{fe:>7.4f}{fmark} {adapt_pred[i]:>9.4f} {ae:>7.4f}{amark}")

    print(f"\n{'─'*78}")
    print("INTERPRETATION")
    print(f"{'─'*78}")
    print("  - Compare adaptive E[R] vs Holt-Winters and SES. If adaptive substantially")
    print("    outperforms, that's evidence P(t)'s formulation is doing more than smoothing.")
    print("    If it ties, Lourie's critique stands and Section 4 needs reframing.")
    print()
    print("  - Optimal α scaling: if α* clearly tracks model size monotonically,")
    print("    that's a meaningful pattern. If it's noisy or flat, the α-tuning story")
    print("    in Paper 3 is post-hoc.")
    print()
    print("  - Per-checkpoint table: where fixed PL fails and adaptive succeeds is")
    print("    the actual visible empirical content. Reviewers will look at this.")


if __name__ == '__main__':
    main()
