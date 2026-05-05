"""
Multi-Family α Sweep — Generalization Test for Paper 3 Finding
===============================================================
Tests whether α=0.6 achieving 0% unreliability generalizes
beyond Pythia to PolyPythias, DataDecide, and OLMo.

Requires evals_all.csv produced by extract_evals.py.
Also runs on the original Pythia data for comparison.

Usage:
    python pt_alpha_sweep_multi.py

Run from C:\\Users\\Carolina\\
"""

import os, json, re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Pythia loader (original, already verified) ────────────────────────
PYTHIA_BASE = './pythia-main/evals/pythia-v1'
PYTHIA_MODELS = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']
PYTHIA_BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande',
                     'arc_easy', 'arc_challenge']

SPLIT = 8
THRESH = 0.02
ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_BOOT = 5000
RNG_SEED = 42


def load_pythia_curve(model):
    folder = os.path.join(PYTHIA_BASE, model, 'zero-shot')
    if not os.path.exists(folder):
        return None
    step_scores = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m: continue
        step = int(m.group(1))
        if step < 1000: continue
        try:
            with open(os.path.join(folder, fname)) as f:
                data = json.load(f)
            r = data.get('results', {})
            sc = []
            for b in PYTHIA_BENCHMARKS:
                if b in r:
                    bd = r[b]
                    s = bd.get('acc_norm', bd.get('acc', None))
                    if s is not None: sc.append(s)
            if len(sc) == len(PYTHIA_BENCHMARKS):
                step_scores[step] = float(np.mean(sc))
        except Exception:
            pass
    steps = sorted(step_scores.keys())
    if len(steps) < 16: return None
    return np.array([step_scores[s] for s in steps])


# ── Power law prior ───────────────────────────────────────────────────

def power_law(x, a, b, c):
    return a * np.power(x, b) + c


def fit_prior(scores, split=SPLIT):
    x_fit = np.arange(1, split + 1, dtype=float)
    x_full = np.arange(1, len(scores) + 1, dtype=float)
    try:
        popt, _ = curve_fit(power_law, x_fit, scores[:split],
                             p0=[0.1, 0.3, 0.3], maxfev=10000,
                             bounds=([-np.inf, -2, 0], [np.inf, 2, 1]))
        return power_law(x_full, *popt)
    except RuntimeError:
        return np.full(len(scores), scores[split - 1])


# ── Predictors ────────────────────────────────────────────────────────

def predict_fixed(scores, split=SPLIT):
    prior = fit_prior(scores, split)
    return prior[split:], scores[split:]


def predict_ema(scores, alpha, split=SPLIT):
    er = scores[0]
    er_series = [er]
    for s in scores[1:]:
        er = (1 - alpha) * er + alpha * s
        er_series.append(er)
    return np.array(er_series[split - 1:-1]), scores[split:]


# ── Metrics ──────────────────────────────────────────────────────────

def unreliability(preds, actuals):
    return float(np.mean(np.abs(preds - actuals) > THRESH))


def bootstrap_ci(errors, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rates = [np.mean(rng.choice(errors, len(errors), replace=True) > THRESH)
             for _ in range(n_boot)]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


# ── Family loader from CSV ────────────────────────────────────────────

def load_family_from_csv(csv_path, family_filter=None, n_checkpoints=16):
    """
    Load curves from the merged CSV output of extract_evals.py.
    Returns: dict of {model_name: np.array of scores}
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    if family_filter:
        df = df[df['family'].str.contains(family_filter, na=False)]

    curves = {}
    for model in df['model'].unique():
        mdf = df[df['model'] == model].sort_values('step')
        scores = mdf['score'].values
        if len(scores) < 12:
            continue
        # Sample n_checkpoints evenly
        indices = np.linspace(0, len(scores) - 1, n_checkpoints, dtype=int)
        curves[model] = scores[indices]

    return curves


# ── Alpha sweep for one family ────────────────────────────────────────

def run_sweep_for_family(curves, family_name, split=SPLIT):
    """Run full alpha sweep for a dict of {model: scores} curves."""
    if not curves:
        return None

    print(f"\n  {family_name} ({len(curves)} configs)")
    print(f"  {'α':<6}", end="")
    for m in curves:
        print(f"  {m.split('-')[-1]:>8}", end="")
    print(f"  {'AGG%':>8} {'95% CI':>16}")
    print("  " + "-" * (6 + 10*len(curves) + 30))

    results = {}
    for alpha in ALPHA_GRID:
        model_rates = []
        all_errs = []
        for model, scores in curves.items():
            if len(scores) < split + 2:
                continue
            preds, acts = predict_ema(scores, alpha)
            errs = np.abs(preds - acts)
            model_rates.append(unreliability(preds, acts))
            all_errs.extend(errs.tolist())

        all_errs = np.array(all_errs)
        agg = float(np.mean(all_errs > THRESH))
        lo, hi = bootstrap_ci(all_errs)
        mark = " ←" if agg <= 0.05 else ""

        print(f"  {alpha:<6.2f}", end="")
        for r in model_rates:
            print(f"  {r*100:>7.0f}%", end="")
        print(f"  {agg*100:>7.0f}% [{lo*100:>4.1f},{hi*100:>4.1f}]{mark}")
        results[alpha] = dict(agg=agg, lo=lo, hi=hi, per_model=model_rates)

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Multi-Family α Sweep — Does α=0.6 generalize beyond Pythia?")
    print("=" * 80)
    print(f"\nThreshold: {THRESH} (absolute error), Split: {SPLIT}/{SPLIT}")
    print(f"Paper 3 finding: α=0.3 → 38% unreliable. α=0.6 → 0%.")
    print(f"Question: does α=0.6 achieve ~0% on other families too?")

    all_family_results = {}

    # ── FAMILY 1: Pythia (reference) ─────────────────────────────────
    print("\n" + "=" * 80)
    print("PYTHIA (reference — Paper 3 result)")
    print("=" * 80)
    pythia_curves = {}
    for m in PYTHIA_MODELS:
        c = load_pythia_curve(m)
        if c is not None:
            pythia_curves[m] = c
    if pythia_curves:
        all_family_results['pythia'] = run_sweep_for_family(
            pythia_curves, 'Pythia')
    else:
        print("  Pythia data not found at", PYTHIA_BASE)

    # ── FAMILY 2: PolyPythias ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("POLYPYTHIAS (EleutherAI — same architecture, different seeds/sizes)")
    print("=" * 80)
    poly_curves = load_family_from_csv('evals_all.csv', 'polypythia')
    if poly_curves:
        all_family_results['polypythia'] = run_sweep_for_family(
            poly_curves, 'PolyPythias')
    else:
        print("  evals_polypythia.csv not found.")
        print("  Run: python extract_evals.py --polypythia")

    # ── FAMILY 3: DataDecide ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DATADECIDE (AI2 — different training recipes, 14 model sizes)")
    print("=" * 80)
    dd_curves = load_family_from_csv('evals_all.csv', 'datadecide')
    if dd_curves:
        all_family_results['datadecide'] = run_sweep_for_family(
            dd_curves, 'DataDecide')
    else:
        print("  evals_datadecide.csv not found.")
        print("  Run: python extract_evals.py --datadecide")

    # ── FAMILY 4: OLMo ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OLMO (AI2 — different architecture, different training data)")
    print("=" * 80)
    olmo_curves = load_family_from_csv('evals_all.csv', 'olmo')
    if olmo_curves:
        all_family_results['olmo'] = run_sweep_for_family(
            olmo_curves, 'OLMo')
    else:
        print("  evals_olmo.csv not found.")
        print("  Run: python extract_evals.py --olmo")

    # ── Summary across families ───────────────────────────────────────
    if len(all_family_results) < 2:
        print("\nNot enough families loaded for summary comparison.")
        print("Run extract_evals.py first, then re-run this script.")
        return

    print("\n" + "=" * 80)
    print("SUMMARY — Unreliability % at each α across families")
    print("=" * 80)
    print(f"\n  {'α':<6}", end="")
    for fam in all_family_results:
        print(f"  {fam:>14}", end="")
    print()
    print("  " + "-" * (6 + 16 * len(all_family_results)))

    for alpha in ALPHA_GRID:
        print(f"  {alpha:<6.2f}", end="")
        any_nonzero = False
        for fam, results in all_family_results.items():
            if results and alpha in results:
                agg = results[alpha]['agg']
                print(f"  {agg*100:>13.0f}%", end="")
                if agg > 0.05: any_nonzero = True
            else:
                print(f"  {'N/A':>14}", end="")
        mark = " ←" if not any_nonzero else ""
        print(mark)

    print("\n  '←' = all families at ≤5% unreliability at this α")

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print("""
  If '←' appears at α=0.6 across all loaded families:
    → α=0.6 is a robust default across architectures and training recipes.
    → Tool claim: "set α=0.6 as default; run alpha_sweep() to calibrate
      to your specific family."
    → Paper 3 follow-up is defensible and publishable.

  If '←' only on Pythia-related families (Pythia + PolyPythias):
    → α=0.6 may be specific to the GPT-NeoX architecture or The Pile data.
    → Tool claim must be softer: "α=0.6 works for Pythia-family models;
      sweep α for other architectures."
    → Still useful, just narrower.

  If different families need different α to reach 0%:
    → The alpha_sweep() utility is the core tool value — automatic
      calibration to your family. α=0.6 is not universal default.

  In all cases, the 96%→0% improvement over fixed power-law holds
  for Pythia. The generalization question changes which version of
  the tool claim we make.
""")


if __name__ == '__main__':
    main()
