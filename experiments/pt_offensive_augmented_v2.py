"""
Paper 3 — Offensive Augmentation (CORRECTED to match original metric)
=======================================================================

Previous script had three bugs:
  1. UNRELIABILITY_THRESH was 0.10 (relative). Paper 3 uses 0.02 (absolute).
  2. Error was relative |pred-actual|/actual. Paper 3 uses absolute |pred-actual|.
  3. Used 7 configs. Paper 3 uses 3: pythia-160m, pythia-1.4b, pythia-6.9b.

This corrects all three. Now testing offensive augmentation against the
PROPER baseline (96% fixed, 38% adaptive EMA from Paper 3's actual code).

Run from C:\\Users\\Carolina\\:
    python pt_offensive_augmented_v2.py
"""

import os, json, re
import numpy as np
from scipy.optimize import curve_fit

BASE = './pythia-main/evals/pythia-v1'
BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge']

# Match Paper 3 exactly
MODELS = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']
SPLIT = 8
UNRELIABILITY_THRESH = 0.02  # ABSOLUTE error threshold
ALPHA_DEFAULT = 0.3


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


# ── Baselines (matching Paper 3 exactly) ─────────────────────────────

def predict_fixed(scores, split=SPLIT):
    """Fixed power-law extrapolation from first `split` checkpoints."""
    prior = fit_power_law_prior(scores, split)
    return prior[split:], scores[split:]


def predict_adaptive_ema(scores, alpha=ALPHA_DEFAULT, split=SPLIT):
    """Pre-update adaptive baseline. E[R](t-1) predicts score(t)."""
    er = scores[0]
    er_series = [er]
    for s in scores[1:]:
        er = (1 - alpha) * er + alpha * s
        er_series.append(er)
    return np.array(er_series[split - 1:-1]), scores[split:]


# ── Offensive variants (augmentation on power-law prior) ─────────────

def predict_offensive_additive(scores, gamma, split=SPLIT):
    """Additive smoothed-ratchet correction on power-law prior."""
    prior = fit_power_law_prior(scores, split)
    smoothed_dev = 0.0

    # Calibrate through fitting region
    for t in range(split):
        dev = scores[t] - prior[t]
        smoothed_dev = (1 - gamma) * smoothed_dev + gamma * dev

    # Predict and update through test region (use pre-update state)
    adjusted = np.zeros(len(scores) - split)
    for i, t in enumerate(range(split, len(scores))):
        adjusted[i] = prior[t] + smoothed_dev
        dev = scores[t] - prior[t]
        smoothed_dev = (1 - gamma) * smoothed_dev + gamma * dev

    return adjusted, scores[split:]


def predict_offensive_multiplicative(scores, gamma, split=SPLIT):
    """Multiplicative smoothed-ratchet correction on power-law prior."""
    prior = fit_power_law_prior(scores, split)
    smoothed_ratio = 1.0

    for t in range(split):
        ratio = scores[t] / max(prior[t], 1e-6)
        smoothed_ratio = (1 - gamma) * smoothed_ratio + gamma * ratio

    adjusted = np.zeros(len(scores) - split)
    for i, t in enumerate(range(split, len(scores))):
        adjusted[i] = prior[t] * smoothed_ratio
        ratio = scores[t] / max(prior[t], 1e-6)
        smoothed_ratio = (1 - gamma) * smoothed_ratio + gamma * ratio

    return adjusted, scores[split:]


# ── Reliability (matching Paper 3 exactly: ABSOLUTE error) ───────────

def compute_metrics(y_pred, y_actual, threshold=UNRELIABILITY_THRESH):
    """Paper 3's exact definition: absolute error > threshold = unreliable."""
    errors = np.abs(y_pred - y_actual)
    return dict(
        mae=float(np.mean(errors)),
        unreliable=float(np.mean(errors > threshold)),
        n=len(errors),
    )


def aggregate_metrics(per_model_results, key):
    """Pool across models for aggregate result."""
    all_preds = []; all_acts = []
    for r in per_model_results.values():
        all_preds.extend(r[f'{key}_preds'])
        all_acts.extend(r['actuals'])
    return compute_metrics(np.array(all_preds), np.array(all_acts))


# ── Experiment ───────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("Paper 3 Offensive Augmentation — CORRECTED to match original metric")
    print("=" * 90)
    print()
    print(f"Configs: {MODELS}")
    print(f"Split: {SPLIT}/{SPLIT}, threshold: {UNRELIABILITY_THRESH} (absolute error)")
    print()

    curves = {}
    for m in MODELS:
        c = load_curve(m)
        if c is not None:
            curves[m] = c
        else:
            print(f"WARNING: could not load {m}")
    print(f"Loaded {len(curves)} configs\n")

    # First, replicate Paper 3's headline result to confirm methodology
    print("=" * 90)
    print("VALIDATION: Replicating Paper 3 headline (should match published numbers)")
    print("=" * 90)
    print()
    print(f"{'Model':<14} {'Fixed MAE':>10} {'Fixed%':>8} {'Adapt MAE':>10} {'Adapt%':>8}")
    print("-" * 60)

    per_model = {}
    for m, scores in curves.items():
        pf, af = predict_fixed(scores)
        pe, ae = predict_adaptive_ema(scores)
        mf = compute_metrics(pf, af)
        me = compute_metrics(pe, ae)
        per_model[m] = dict(
            actuals=af.tolist(),
            fixed_preds=pf.tolist(),
            adapt_preds=pe.tolist(),
        )
        print(f"{m:<14} {mf['mae']:>10.4f} {mf['unreliable']*100:>7.0f}% "
              f"{me['mae']:>10.4f} {me['unreliable']*100:>7.0f}%")

    # Aggregate
    fixed_agg = aggregate_metrics(per_model, 'fixed')
    adapt_agg = aggregate_metrics(per_model, 'adapt')
    print(f"{'AGGREGATE':<14} {fixed_agg['mae']:>10.4f} "
          f"{fixed_agg['unreliable']*100:>7.0f}% "
          f"{adapt_agg['mae']:>10.4f} "
          f"{adapt_agg['unreliable']*100:>7.0f}%")

    if abs(fixed_agg['unreliable'] - 0.96) < 0.10:
        print("\n  ✓ Fixed unreliability matches Paper 3 (~96%)")
    else:
        print(f"\n  ✗ Fixed unreliability ({fixed_agg['unreliable']*100:.0f}%) "
              f"doesn't match Paper 3 (96%) — investigate before trusting offensive")

    if abs(adapt_agg['unreliable'] - 0.38) < 0.15:
        print(f"  ✓ Adaptive EMA unreliability ({adapt_agg['unreliable']*100:.0f}%) "
              f"~matches Paper 3 (~38%)")
    else:
        print(f"  ✗ Adaptive EMA unreliability ({adapt_agg['unreliable']*100:.0f}%) "
              f"doesn't match Paper 3 (38%)")

    # Now test offensive variants
    print()
    print("=" * 90)
    print("OFFENSIVE FRAMINGS — γ sweep (compare to defensive EMA above)")
    print("=" * 90)
    print()
    print(f"Bar to beat: {adapt_agg['unreliable']*100:.0f}% (adaptive EMA)")
    print(f"Improvement on: {fixed_agg['unreliable']*100:.0f}% (fixed power-law)")
    print()

    gamma_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    print(f"{'Framing':<22} {'γ':<6} {'MAE':>10} {'unrel%':>9} "
          f"{'Δ vs EMA':>10} {'Δ vs fixed':>11}")
    print("-" * 80)

    for framing_name, fn in [
        ('Offensive_A_mult', predict_offensive_multiplicative),
        ('Offensive_B_add',  predict_offensive_additive),
    ]:
        for gamma in gamma_grid:
            preds_by_model = {}
            for m, scores in curves.items():
                p, a = fn(scores, gamma=gamma)
                preds_by_model[m] = dict(actuals=a.tolist(),
                                          off_preds=p.tolist())
            agg = aggregate_metrics(preds_by_model, 'off')
            d_ema = (agg['unreliable'] - adapt_agg['unreliable']) * 100
            d_fix = (agg['unreliable'] - fixed_agg['unreliable']) * 100
            print(f"{framing_name:<22} {gamma:<6.2f} "
                  f"{agg['mae']:>10.4f} {agg['unreliable']*100:>8.0f}% "
                  f"{d_ema:>+9.0f}% {d_fix:>+10.0f}%")
        print()

    # Find best gamma for each framing on aggregate unreliability
    best = {'a': (None, 1.0), 'b': (None, 1.0)}
    for gamma in gamma_grid:
        preds_a = {}; preds_b = {}
        for m, scores in curves.items():
            pa, a = predict_offensive_multiplicative(scores, gamma=gamma)
            pb, _ = predict_offensive_additive(scores, gamma=gamma)
            preds_a[m] = dict(actuals=a.tolist(), x_preds=pa.tolist())
            preds_b[m] = dict(actuals=a.tolist(), x_preds=pb.tolist())
        u_a = aggregate_metrics(preds_a, 'x')['unreliable']
        u_b = aggregate_metrics(preds_b, 'x')['unreliable']
        if u_a < best['a'][1]: best['a'] = (gamma, u_a)
        if u_b < best['b'][1]: best['b'] = (gamma, u_b)

    print("=" * 90)
    print(f"Per-model breakdown — Offensive A (γ={best['a'][0]}) "
          f"vs Offensive B (γ={best['b'][0]}) vs EMA")
    print("=" * 90)
    print(f"\n{'Model':<14} {'Fixed%':>8} {'EMA%':>7} "
          f"{'OffA%':>7} {'OffB%':>7}")
    print("-" * 50)
    for m, scores in curves.items():
        pf, af = predict_fixed(scores)
        pe, _ = predict_adaptive_ema(scores)
        pa, _ = predict_offensive_multiplicative(scores, gamma=best['a'][0])
        pb, _ = predict_offensive_additive(scores, gamma=best['b'][0])
        print(f"{m:<14} "
              f"{compute_metrics(pf, af)['unreliable']*100:>7.0f}% "
              f"{compute_metrics(pe, af)['unreliable']*100:>6.0f}% "
              f"{compute_metrics(pa, af)['unreliable']*100:>6.0f}% "
              f"{compute_metrics(pb, af)['unreliable']*100:>6.0f}%")

    # Trajectory inspection at best gamma for one config
    if 'pythia-1.4b' in curves:
        scores = curves['pythia-1.4b']
        prior = fit_power_law_prior(scores)
        pe, _ = predict_adaptive_ema(scores)
        pa, _ = predict_offensive_multiplicative(scores, gamma=best['a'][0])
        pb, _ = predict_offensive_additive(scores, gamma=best['b'][0])
        print()
        print("=" * 90)
        print(f"Trajectory inspection — pythia-1.4b "
              f"(OffA γ={best['a'][0]}, OffB γ={best['b'][0]}, threshold={UNRELIABILITY_THRESH})")
        print("=" * 90)
        print(f"\n{'Ckpt':>4} {'Actual':>8} {'Fixed':>8} {'|err|':>7} "
              f"{'EMA':>8} {'|err|':>7} {'OffA':>8} {'|err|':>7} "
              f"{'OffB':>8} {'|err|':>7}")
        print("-" * 90)
        for i in range(len(scores) - SPLIT):
            t = SPLIT + i
            actual = scores[t]
            f_err = abs(prior[t] - actual)
            e_err = abs(pe[i] - actual)
            a_err = abs(pa[i] - actual)
            b_err = abs(pb[i] - actual)
            f_mark = '!' if f_err > UNRELIABILITY_THRESH else ' '
            e_mark = '!' if e_err > UNRELIABILITY_THRESH else ' '
            a_mark = '!' if a_err > UNRELIABILITY_THRESH else ' '
            b_mark = '!' if b_err > UNRELIABILITY_THRESH else ' '
            print(f"{t+1:>4} {actual:>8.4f} "
                  f"{prior[t]:>8.4f} {f_err:>6.4f}{f_mark} "
                  f"{pe[i]:>8.4f} {e_err:>6.4f}{e_mark} "
                  f"{pa[i]:>8.4f} {a_err:>6.4f}{a_mark} "
                  f"{pb[i]:>8.4f} {b_err:>6.4f}{b_mark}")
        print("\n  ! = exceeds 0.02 absolute error threshold (counts as unreliable)")

    print()
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"""
  Bar to beat (adaptive EMA):   {adapt_agg['unreliable']*100:.0f}% unreliability,
                                 {adapt_agg['mae']:.4f} MAE
  Best Offensive A (γ={best['a'][0]}):       {best['a'][1]*100:.0f}% unreliability
  Best Offensive B (γ={best['b'][0]}):       {best['b'][1]*100:.0f}% unreliability

  Read the gamma sweep table:

  STRONG positive: best offensive variant beats EMA's unreliability rate
    → explicit offensive augmentation outperforms implicit version

  COMPARABLE (within ~5%): augmentation matches EMA
    → both are doing the same work, "Paper 3 was secretly offensive"
       framing confirmed but no performance gain

  WORSE: offensive stays above EMA across all γ
    → augmentation introduces lag/noise that hurts; defensive remains
       the right tool for this regime
""")


if __name__ == '__main__':
    main()
