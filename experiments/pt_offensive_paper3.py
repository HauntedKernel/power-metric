"""
P(t) Offensive Variant on Paper 3 — Scaling Reliability with Power-Law Prior
==============================================================================

The setup:
  Paper 3 evaluates whether models scale as predicted by a power law fit
  to early checkpoints. We previously showed that the *adaptive* baseline
  (defensive P(t)) reduces unreliability from 96% to 38% vs the *fixed*
  power-law baseline.

  But that test conflated two things: defensive P(t) wasn't really being
  tested AGAINST the prior — it was REPLACING the prior with an EMA.
  
  The offensive variant uses the power-law fit AS the prior anchor and
  measures how much actual scores exceed (or fall short of) prediction.
  This is the original trading intuition applied to scaling: fixed
  rational expectation, signal measured against it.

  Long horizon (16 checkpoints), real ML data, naturally prior-anchored
  problem. The cleanest test we have for the offensive variant.

Three things this script tests:
  1. STANDALONE OFFENSIVE: Does P(t) with fixed power-law prior detect
     scaling unreliability as well as or better than defensive P(t)?
  
  2. CUSUM TUNING SWEEP: Across (k, h) parameter grid, find calibrations
     where CUSUM correctly classifies regimes (sustained outperformance =
     offensive regime; sustained underperformance = defensive regime).
  
  3. META CONTROLLER: With tuned CUSUM, does the regime-gated meta P(t)
     beat either standalone variant on real data?

Run from C:\\Users\\Carolina\\:
    python pt_offensive_paper3.py
"""

import os, json, re
import numpy as np
from scipy.optimize import curve_fit
from itertools import product

# ── Data loading (same as Paper 3 verification) ──────────────────────

BASE = './pythia-main/evals/pythia-v1'
BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']
TARGET_MODELS = ['pythia-70m','pythia-160m','pythia-410m',
                 'pythia-1.4b','pythia-2.8b','pythia-6.9b','pythia-12b']
SPLIT = 8  # first 8 fit prior, last 8 test

ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3


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
    """Fit power law to first `split` checkpoints. Return prediction for full curve."""
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


# ── P(t) variants ────────────────────────────────────────────────────

def pt_defensive(scores, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """V1 EWMA. E(t) = s/R(t-1), R adapts."""
    er = None; ew = 0.0; pw = 0.0
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        if er is None:
            eff = 1.0; er = max(s, 1e-6)
        else:
            eff = s / er
            er = (1 - alpha) * er + alpha * s
        win = 1.0 if eff >= 1.0 else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


def pt_offensive(scores, prior_curve,
                  alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """
    Offensive variant. E(t) = s(t) / prior(t), where prior is the
    fitted power-law prediction at checkpoint t. Smoothed efficiency
    to reduce per-step volatility.
    """
    eff_smoothed = None; ew = 0.0; pw = 0.0
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s); p = max(float(prior_curve[t]), 1e-6)
        eff_raw = s / p
        if eff_smoothed is None:
            eff_smoothed = eff_raw
        else:
            eff_smoothed = (1 - alpha) * eff_smoothed + alpha * eff_raw
        win = 1.0 if eff_smoothed >= 1.0 else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff_smoothed * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


def cusum_regime_trace(scores, prior_curve, k=0.005, h=0.05, warmup=2):
    """
    Run CUSUM regime detection on the deviation s(t) - prior(t).
    Returns array of regime labels: 'defensive' or 'offensive' per timestep.

    Defaults are smaller than the synthetic version because Pythia score
    deviations from power-law are small in absolute terms.
    """
    cusum_up = 0.0; cusum_down = 0.0; regime = 'defensive'
    out = []
    for t, (s, p) in enumerate(zip(scores, prior_curve)):
        deviation = float(s) - float(p)
        cusum_up = max(0.0, cusum_up + deviation - k)
        cusum_down = max(0.0, cusum_down - deviation - k)
        if t < warmup:
            out.append('defensive'); continue
        if regime == 'defensive':
            if cusum_up > h:
                regime = 'offensive'; cusum_up = 0.0
        else:
            if cusum_down > h:
                regime = 'defensive'; cusum_down = 0.0
        out.append(regime)
    return out


def pt_meta(scores, prior_curve, k_cusum=0.005, h_cusum=0.05):
    """Meta-controller: select active P(t) by CUSUM regime."""
    def_pt = pt_defensive(scores)
    off_pt = pt_offensive(scores, prior_curve)
    regimes = cusum_regime_trace(scores, prior_curve, k=k_cusum, h=h_cusum)
    active = np.array([off_pt[i] if r == 'offensive' else def_pt[i]
                        for i, r in enumerate(regimes)])
    return active, regimes, def_pt, off_pt


# ── Reliability metrics ──────────────────────────────────────────────

UNRELIABILITY_THRESH = 0.10  # Paper 3 default: prediction off by >10% = unreliable


def reliability_against_curve(predicted, actual):
    """
    Return:
        unreliable_rate: fraction of timesteps where |predicted - actual|/actual > 10%
        mae:             mean absolute error
    """
    err = np.abs(predicted - actual) / np.maximum(actual, 1e-6)
    return float(np.mean(err > UNRELIABILITY_THRESH)), float(np.mean(np.abs(predicted - actual)))


def pt_signal_classifies_unreliable(pt_signal, predicted, actual,
                                       threshold_low=0.5):
    """
    Treat P(t) signal as 'unreliability detector':
    For each post-split timestep, if P(t) is low, the prior was wrong
    (signal underperforming prediction). This should correlate with high
    actual error.

    Returns:
        precision, recall on the binary task: did P(t) drop below threshold
        when actual error exceeded 10%?
    """
    n = len(pt_signal)
    actual_unrel = (np.abs(predicted - actual) / np.maximum(actual, 1e-6)
                     > UNRELIABILITY_THRESH)
    pt_low = pt_signal < threshold_low
    tp = np.sum(actual_unrel & pt_low)
    fp = np.sum((~actual_unrel) & pt_low)
    fn = np.sum(actual_unrel & (~pt_low))
    tn = np.sum((~actual_unrel) & (~pt_low))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, dict(tp=tp, fp=fp, fn=fn, tn=tn)


# ── Main experiment ──────────────────────────────────────────────────

def run_per_model_analysis():
    """Run defensive vs offensive vs meta for each Pythia model."""
    print("="*90)
    print("Per-Model Analysis — Defensive vs Offensive vs Meta P(t)")
    print("="*90)

    curves = {}
    for m in TARGET_MODELS:
        c = load_curve(m)
        if c is not None:
            curves[m] = c
    print(f"\nLoaded {len(curves)} configs, {len(next(iter(curves.values())))} ckpts each\n")

    # Headline table
    print(f"{'Model':<14} {'unrel%':>8} {'def_pt@last':>12} {'off_pt@last':>12} "
          f"{'meta_pt@last':>13} {'meta switches':>14}")
    print("-" * 90)
    for m, scores in curves.items():
        prior = fit_power_law_prior(scores)
        # Reliability is computed only on post-split portion (the test region)
        unrel, mae = reliability_against_curve(prior[SPLIT:], scores[SPLIT:])
        def_pt = pt_defensive(scores)
        off_pt = pt_offensive(scores, prior)
        meta_active, regimes, _, _ = pt_meta(scores, prior)
        n_switches = sum(1 for i in range(1, len(regimes))
                         if regimes[i] != regimes[i-1])
        print(f"{m:<14} {unrel*100:>7.1f}% "
              f"{def_pt[-1]:>12.4f} {off_pt[-1]:>12.4f} "
              f"{meta_active[-1]:>13.4f} {n_switches:>14}")


def run_cusum_tuning_sweep():
    """Sweep CUSUM (k, h) parameters and measure regime classification quality."""
    print("\n" + "="*90)
    print("CUSUM Tuning Sweep — finding calibrations that produce sensible regime structure")
    print("="*90)

    curves = {m: load_curve(m) for m in TARGET_MODELS}
    curves = {m: c for m, c in curves.items() if c is not None}
    priors = {m: fit_power_law_prior(c) for m, c in curves.items()}

    # Identify "outperforming" configs (large models) vs "underperforming"
    # (small models) based on actual reliability
    config_unrel = {}
    for m, c in curves.items():
        u, _ = reliability_against_curve(priors[m][SPLIT:], c[SPLIT:])
        config_unrel[m] = u

    # Sweep CUSUM grid
    k_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
    h_grid = [0.02, 0.05, 0.10, 0.20, 0.40]

    print(f"\n{'k':<8} {'h':<8} {'avg_switches':>14} {'config_diff':>14} "
          f"{'desc':>30}")
    print("-" * 90)
    print("(config_diff = how distinct configs' final regimes are; higher = better classification)\n")

    for k, h in product(k_grid, h_grid):
        switches = []
        final_regimes = {}
        for m, scores in curves.items():
            regimes = cusum_regime_trace(scores, priors[m], k=k, h=h)
            n_sw = sum(1 for i in range(1, len(regimes))
                       if regimes[i] != regimes[i-1])
            switches.append(n_sw)
            final_regimes[m] = regimes[-1]

        avg_sw = np.mean(switches)
        # config_diff: how many distinct (model, final_regime) splits we see
        n_off = sum(1 for r in final_regimes.values() if r == 'offensive')
        n_def = len(final_regimes) - n_off
        config_diff = min(n_off, n_def)  # 0 = all same; high = balanced

        desc = ""
        if avg_sw < 0.5: desc = "all stay defensive (h too high?)"
        elif avg_sw > 6: desc = "flapping (h/k too low)"
        elif config_diff >= 2: desc = "BALANCED — configs differ by regime"
        else: desc = "all same final regime"

        print(f"{k:<8.4f} {h:<8.4f} {avg_sw:>14.2f} {config_diff:>14d} "
              f"{desc:>30}")

    # Show breakdown for one promising calibration
    print("\n" + "="*90)
    print("Detailed breakdown at k=0.005, h=0.05 (default candidate)")
    print("="*90)
    print(f"\n{'Model':<14} {'actual unrel%':>14} {'final regime':>14} "
          f"{'#switches':>10}")
    print("-" * 60)
    for m, scores in curves.items():
        regimes = cusum_regime_trace(scores, priors[m], k=0.005, h=0.05)
        n_sw = sum(1 for i in range(1, len(regimes))
                   if regimes[i] != regimes[i-1])
        print(f"{m:<14} {config_unrel[m]*100:>13.1f}% "
              f"{regimes[-1]:>14} {n_sw:>10}")


def run_pt_signal_classification():
    """
    Treat the active P(t) signal as a binary classifier for unreliability.
    Compare defensive, offensive, and meta on precision/recall.
    """
    print("\n" + "="*90)
    print("P(t) Signal as Unreliability Classifier")
    print("="*90)
    print("\nFor each timestep in the post-split region (test region),")
    print("does P(t) < 0.5 correctly identify when actual deviates >10% from prior?")
    print()

    curves = {m: load_curve(m) for m in TARGET_MODELS}
    curves = {m: c for m, c in curves.items() if c is not None}

    # Aggregate post-split signals across all configs
    def_signals_post = []
    off_signals_post = []
    meta_signals_post = []
    actuals_post = []
    priors_post = []

    for m, scores in curves.items():
        prior = fit_power_law_prior(scores)
        def_pt = pt_defensive(scores)
        off_pt = pt_offensive(scores, prior)
        meta_active, _, _, _ = pt_meta(scores, prior, k_cusum=0.005, h_cusum=0.05)

        def_signals_post.extend(def_pt[SPLIT:])
        off_signals_post.extend(off_pt[SPLIT:])
        meta_signals_post.extend(meta_active[SPLIT:])
        actuals_post.extend(scores[SPLIT:])
        priors_post.extend(prior[SPLIT:])

    def_signals_post = np.array(def_signals_post)
    off_signals_post = np.array(off_signals_post)
    meta_signals_post = np.array(meta_signals_post)
    actuals_post = np.array(actuals_post)
    priors_post = np.array(priors_post)

    print(f"{'Variant':<12} {'threshold':>10} {'precision':>12} {'recall':>10} "
          f"{'F1':>8} {'tp':>5} {'fp':>5} {'fn':>5}")
    print("-" * 80)
    for variant, sig in [('defensive', def_signals_post),
                          ('offensive', off_signals_post),
                          ('meta',      meta_signals_post)]:
        for theta in [0.3, 0.5, 0.7]:
            p, r, m = pt_signal_classifies_unreliable(sig, priors_post,
                                                       actuals_post,
                                                       threshold_low=theta)
            f1 = 2 * p * r / max(p + r, 1e-6)
            print(f"{variant:<12} {theta:>10.2f} {p:>12.3f} {r:>10.3f} "
                  f"{f1:>8.3f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")


def main():
    print("="*90)
    print("Paper 3 Offensive Variant — Real Pythia Data, Power-Law Prior")
    print("="*90)
    print()
    print("PRIOR: power-law fit to first 8 checkpoints, predicts forward")
    print("DEFENSIVE: classic P(t) with adaptive baseline (replaces prior with EMA)")
    print("OFFENSIVE: P(t) with fitted power-law as fixed prior (original framing)")
    print("META: CUSUM-gated switch between defensive and offensive")
    print()
    print("Test 1: how each variant evaluates each Pythia config at final checkpoint")
    print("Test 2: CUSUM (k, h) parameter sweep — find regime-coherent calibrations")
    print("Test 3: P(t) as unreliability classifier — precision/recall on real data")

    run_per_model_analysis()
    run_cusum_tuning_sweep()
    run_pt_signal_classification()

    print("\n" + "="*90)
    print("INTERPRETATION GUIDE")
    print("="*90)
    print("""
  TEST 1 (per-model): the defensive vs offensive vs meta P(t) values at last
  checkpoint should differ meaningfully if the variants are seeing different
  signal. If they're all similar, the variants are functionally equivalent on
  this problem and there's no fork to exploit.

  TEST 2 (CUSUM tuning): we want to find calibrations where small models
  (which actually scale unreliably) end up in defensive regime AND large
  models (which scale reliably and exceed power-law fit) end up in offensive
  regime. Different final regimes for different configs = CUSUM is doing real
  classification work. All same final regime = CUSUM is degenerate.

  TEST 3 (classification): does the offensive variant's P(t) signal predict
  unreliability events better than defensive? F1 score is the headline.
  If offensive F1 > defensive F1, we have validation that prior-anchored
  framing is more informative on this regime. If meta F1 > both, the
  synthesis is real.

  STRONG positive signal across all three tests → reorganize the catalog
  around offensive/defensive fork. Major framework refinement.

  STRONG only on Test 3 → offensive variant is a useful classifier even
  if standalone numbers look similar. Catalog gets new entry.

  NEGATIVE → power-law-prior framing doesn't strengthen Paper 3. Defensive
  is still the right tool for this problem. Offensive lane stays
  hypothetical pending another long-horizon test.
""")


if __name__ == '__main__':
    main()
