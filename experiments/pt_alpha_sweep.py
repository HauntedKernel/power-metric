"""
Paper 3 — Full α Sweep Across All Models
=========================================
Paper 3's original sweep only tested α in [0.1..0.5] on pythia-1.4b alone.
Our checks showed EMA α=0.5 gets 4% and α=0.7 gets 0% across 3 configs.

This script extends the sweep to:
  - All three Paper 3 models (pythia-160m, pythia-1.4b, pythia-6.9b)
  - Extended α range [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  - Per-model AND aggregate unreliability
  - Bootstrap CIs to show what's real vs noise

The key question: was α=0.3 chosen because higher α genuinely fails on
some configs, or did Paper 3 just not sweep far enough to see α=0.5 win?

If α=0.7 genuinely fails on one config but wins on others, there's a
tradeoff that justifies α=0.3 as a conservative choice. If α=0.7 wins
everywhere, Paper 3 left the result on the table.

Run from C:\\Users\\Carolina\\:
    python pt_alpha_sweep.py
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
THRESH = 0.02
N_BOOT = 5000
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


def predict_ema(scores, alpha, split=SPLIT):
    er = scores[0]
    er_series = [er]
    for s in scores[1:]:
        er = (1 - alpha) * er + alpha * s
        er_series.append(er)
    return np.array(er_series[split - 1:-1]), scores[split:]


def unreliability(preds, actuals, threshold=THRESH):
    return float(np.mean(np.abs(preds - actuals) > threshold))


def bootstrap_ci(errs, threshold=THRESH, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rates = [np.mean(rng.choice(errs, len(errs), replace=True) > threshold)
             for _ in range(n_boot)]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


# ── Sweep ────────────────────────────────────────────────────────────

def run_sweep(model_list, alpha_grid, label=""):
    curves = {}
    for m in model_list:
        c = load_curve(m)
        if c is not None:
            curves[m] = c

    if not curves:
        print("  No data loaded.")
        return

    # Header
    model_cols = "".join(f"{m.replace('pythia-',''):>9}" for m in curves)
    print(f"\n  {'α':<6} {model_cols} {'AGGREGATE':>10} {'95% CI':>18}")
    print("  " + "-" * (6 + 9*len(curves) + 30))

    for alpha in alpha_grid:
        per_model_rates = []
        all_errors = []

        for m, scores in curves.items():
            preds, actuals = predict_ema(scores, alpha)
            errs = np.abs(preds - actuals)
            rate = float(np.mean(errs > THRESH))
            per_model_rates.append(rate)
            all_errors.extend(errs.tolist())

        all_errors = np.array(all_errors)
        agg_rate = float(np.mean(all_errors > THRESH))
        lo, hi = bootstrap_ci(all_errors)

        model_str = "".join(f"{r*100:>8.0f}%" for r in per_model_rates)
        marker = " ←" if agg_rate <= 0.05 else ""
        print(f"  {alpha:<6.2f} {model_str} {agg_rate*100:>9.0f}% "
              f"[{lo*100:>5.1f},{hi*100:>5.1f}]{marker}")


def compare_to_offensive(model_list, alpha_grid):
    """
    Side-by-side: best EMA alpha vs offensive γ=0.5 per model.
    This answers: is offensive ever better than best-tuned EMA?
    """
    from pt_offensive_checks import predict_off_mult, predict_off_add

    curves = {m: load_curve(m) for m in model_list if load_curve(m) is not None}

    print(f"\n  {'Model':<14}", end="")
    for alpha in [0.3, 0.5, 0.7]:
        print(f"  {'EMA α='+str(alpha):>12}", end="")
    print(f"  {'OffA γ=0.5':>12} {'OffB γ=0.5':>12}")
    print("  " + "-" * 80)

    for m, scores in curves.items():
        print(f"  {m:<14}", end="")
        for alpha in [0.3, 0.5, 0.7]:
            preds, acts = predict_ema(scores, alpha)
            r = unreliability(preds, acts)
            print(f"  {r*100:>11.0f}%", end="")
        pa, acts = predict_off_mult(scores, gamma=0.5)
        pb, _ = predict_off_add(scores, gamma=0.5)
        print(f"  {unreliability(pa, acts)*100:>11.0f}%"
              f"  {unreliability(pb, acts)*100:>11.0f}%")


def main():
    alpha_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ── Part 1: Paper 3 subset ────────────────────────────────────────
    print("=" * 90)
    print("Paper 3 α Sweep — EMA Adaptive Baseline")
    print("=" * 90)

    print("\n[Paper 3 subset: pythia-160m, pythia-1.4b, pythia-6.9b]")
    print("(Paper 3's original sweep only ran α=0.1-0.5 on pythia-1.4b alone)")
    run_sweep(PAPER3_MODELS, alpha_grid)

    # ── Part 2: Full 7-config ─────────────────────────────────────────
    print("\n[Full 7-config Pythia set]")
    run_sweep(ALL_MODELS, alpha_grid)

    # ── Part 3: Per-model detail at high α ────────────────────────────
    print()
    print("=" * 90)
    print("Per-model detail — does high α fail on any specific config?")
    print("=" * 90)

    curves = {m: load_curve(m) for m in ALL_MODELS if load_curve(m) is not None}

    print(f"\n  {'Model':<14}", end="")
    for alpha in alpha_grid:
        print(f"  {f'α={alpha}':>7}", end="")
    print()
    print("  " + "-" * (14 + 9 * len(alpha_grid) + 5))

    for m, scores in curves.items():
        print(f"  {m:<14}", end="")
        for alpha in alpha_grid:
            preds, acts = predict_ema(scores, alpha)
            r = unreliability(preds, acts)
            flag = "!" if r > 0.50 else (" " if r > 0.0 else "✓")
            print(f"  {r*100:>5.0f}%{flag}", end="")
        print()

    print()
    print("  '✓' = 0% unreliability at this α")
    print("  '!' = >50% unreliability — worse than moderate α")

    # ── Part 4: Why did Paper 3 choose α=0.3? ────────────────────────
    print()
    print("=" * 90)
    print("Trajectory inspection — does high α introduce new problems?")
    print("(Compare α=0.3 vs α=0.7 on pythia-6.9b, the hardest config)")
    print("=" * 90)

    m = 'pythia-6.9b'
    scores = load_curve(m)
    if scores is not None:
        prior = fit_power_law(scores)
        print(f"\n  {'Ckpt':>4} {'Actual':>8} {'Fixed':>8} "
              f"{'EMA α=0.3':>11} {'err':>7} {'EMA α=0.5':>11} {'err':>7} "
              f"{'EMA α=0.7':>11} {'err':>7}")
        print("  " + "-" * 90)

        p3, a3 = predict_ema(scores, 0.3)
        p5, a5 = predict_ema(scores, 0.5)
        p7, a7 = predict_ema(scores, 0.7)

        for i in range(len(scores) - SPLIT):
            t = SPLIT + i
            actual = scores[t]
            e3 = abs(p3[i] - actual)
            e5 = abs(p5[i] - actual)
            e7 = abs(p7[i] - actual)
            m3 = "!" if e3 > THRESH else " "
            m5 = "!" if e5 > THRESH else " "
            m7 = "!" if e7 > THRESH else " "
            print(f"  {t+1:>4} {actual:>8.4f} {prior[t]:>8.4f} "
                  f"{p3[i]:>11.4f} {e3:>6.4f}{m3} "
                  f"{p5[i]:>11.4f} {e5:>6.4f}{m5} "
                  f"{p7[i]:>11.4f} {e7:>6.4f}{m7}")
        print()
        print("  ! = exceeds 0.02 threshold")

    # ── Part 5: The verdict ───────────────────────────────────────────
    print()
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print("""
  KEY QUESTIONS THIS SWEEP ANSWERS:

  1. Does high α (0.5-0.7) consistently beat α=0.3 across ALL configs?
     → If yes: Paper 3 understated its own result. The defensible headline
               is much stronger than published.
     → If no (some config gets worse at high α): α=0.3 was a conservative
               choice to avoid the worst-case. The tradeoff was real.

  2. Is there a single α that achieves 0% unreliability across all configs?
     → If yes: simple, clean, publishable improvement to Paper 3.
     → If no (some configs need low α, some need high): there's a per-config
               optimization story worth telling.

  3. Is there any α where offensive γ=0.5 beats the best EMA?
     → If yes: offensive has real value even at optimal EMA baseline.
     → If no: offensive was never adding something EMA couldn't do with
               better tuning. Not a failure — just clarifies mechanism.

  The per-config detail table (Part 3) is the most important output.
  Look for '!' marks at high α — those are the configs where aggressive
  fast-tracking introduces new failures.
""")


if __name__ == '__main__':
    main()
