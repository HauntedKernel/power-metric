"""
Sections 4, 5, 6 Verification — All Pythia Analyses
====================================================
Single script covering:
  - Section 4: Scaling law reliability (Paper 3) — fixed PL vs adaptive E[R],
               α sensitivity, MAE comparisons, bootstrap CIs
  - Section 5: Per-domain health monitoring (Paper 4) — per-domain P_i,
               correlations with bootstrap CIs (n=5 is small),
               cross-scale Spearman of allocations (the actual claim)
  - Section 6: Training health monitoring (Paper 1) — candidate fractions
               at thresholds, bootstrap CIs over checkpoints

Run from C:\\Users\\Carolina\\:
    python pythia_verifications.py
"""

import os, json, re
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from scipy.optimize import curve_fit

ALPHA       = 0.3
LAMBDA      = 0.5
EWMA_SPAN   = 3
RNG_SEED    = 42
N_BOOTSTRAP = 5000

BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']
DOMAIN_NAMES = {
    'lambada_openai': 'Language/Web',
    'arc_easy':       'Science/Facts',
    'piqa':           'Physical Comm.',
    'arc_challenge':  'Reasoning',
    'winogrande':     'Social Comm.',
}
MODELS = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']
BASE   = './pythia-main/evals/pythia-v1'


# ── Loaders ─────────────────────────────────────────────────────────────

def load_composite(model):
    """Composite score = mean across 5 benchmarks per checkpoint."""
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


def load_per_domain(model):
    folder = os.path.join(BASE, model, 'zero-shot')
    step_dom = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m: continue
        step = int(m.group(1))
        if step < 1000: continue
        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)
        r = data.get('results', {})
        row = {}
        for b in BENCHMARKS:
            if b in r:
                bd = r[b]
                s = bd.get('acc_norm', bd.get('acc', None))
                if s is not None:
                    row[b] = s
        if len(row) == len(BENCHMARKS):
            step_dom[step] = row
    steps = sorted(step_dom.keys())
    return steps, step_dom


# ── P(t) ────────────────────────────────────────────────────────────────

def compute_pm(scores, alpha=ALPHA):
    er=None; ew=0.0; pw=0.0; powers=[]
    for s in scores:
        if er is None:
            eff=1.0; er=max(s,1e-6)
        else:
            eff=s/er
            er=(1-alpha)*er + alpha*s
        win = 1.0 if eff>1.0 else 0.0
        a = 2.0/(EWMA_SPAN+1)
        ew = a*win + (1-a)*ew
        inst = eff*ew
        pw = np.exp(-LAMBDA)*pw + (1-np.exp(-LAMBDA))*inst
        powers.append(pw)
    return np.array(powers)


def adaptive_baseline(scores, alpha=ALPHA):
    """Pre-update E[R] series — what Paper 3 uses as adaptive predictor."""
    er = None
    series = []
    for s in scores:
        prev = er if er is not None else s
        series.append(prev)
        er = s if er is None else (1-alpha)*er + alpha*s
    return np.array(series)


# ── Section 4: Scaling reliability ──────────────────────────────────────

def power_law(x, a, b, c):
    return a * np.power(x, b) + c


def section4_analysis(threshold=0.02, alphas=(0.1, 0.2, 0.3, 0.4, 0.5)):
    print("\n" + "="*78)
    print("SECTION 4 — Scaling Law Reliability (Paper 3 replication)")
    print("="*78)

    rng = np.random.default_rng(RNG_SEED)

    for model in MODELS:
        steps, scores = load_composite(model)
        n = len(scores)
        n_fit = n // 2
        x = np.arange(1, n+1)

        # Fixed power-law fit on first half
        try:
            popt, _ = curve_fit(power_law, x[:n_fit], scores[:n_fit],
                                p0=[1.0, -0.1, 0.5], maxfev=20000)
            fixed_pred = power_law(x[n_fit:], *popt)
        except Exception:
            fixed_pred = np.full(n - n_fit, scores[:n_fit].mean())

        actual = scores[n_fit:]
        fixed_err = np.abs(fixed_pred - actual)
        fixed_unrel = (fixed_err > threshold).mean()
        fixed_mae = fixed_err.mean()

        # Bootstrap CIs on fixed unreliability
        boot_unrel = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(actual), len(actual))
            boot_unrel.append((fixed_err[idx] > threshold).mean())
        f_lo, f_hi = np.percentile(boot_unrel, [2.5, 97.5])

        print(f"\n  {model}  (n={n} ckpts, fit={n_fit}, predict={n-n_fit})")
        print(f"    Fixed PL : MAE={fixed_mae:.4f}  unrel={fixed_unrel*100:.0f}%  "
              f"95% CI [{f_lo*100:.0f}, {f_hi*100:.0f}]%")

        # Adaptive at multiple alphas
        for a in alphas:
            adapt = adaptive_baseline(scores, alpha=a)
            adapt_pred = adapt[n_fit:]
            adapt_err = np.abs(adapt_pred - actual)
            adapt_unrel = (adapt_err > threshold).mean()
            adapt_mae = adapt_err.mean()

            boot = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, len(actual), len(actual))
                boot.append((adapt_err[idx] > threshold).mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])

            tag = '  ←' if a == 0.3 else ''
            print(f"    α={a:.1f}    : MAE={adapt_mae:.4f}  "
                  f"unrel={adapt_unrel*100:.0f}%  "
                  f"95% CI [{lo*100:.0f}, {hi*100:.0f}]%{tag}")


# ── Section 5: Per-domain ───────────────────────────────────────────────

def section5_analysis():
    print("\n" + "="*78)
    print("SECTION 5 — Per-Domain Health (Paper 4 replication + cross-scale test)")
    print("="*78)

    rng = np.random.default_rng(RNG_SEED)
    final_props_by_model = {}
    final_scores_by_model = {}

    for model in MODELS:
        steps, step_dom = load_per_domain(model)
        all_powers = {b: [] for b in BENCHMARKS}
        per_dom_state = {b: {'er': None, 'ew': 0.0, 'pw': 0.0} for b in BENCHMARKS}

        for step in steps:
            for b in BENCHMARKS:
                s = step_dom[step][b]
                st = per_dom_state[b]
                if st['er'] is None:
                    eff = 1.0; st['er'] = max(s, 1e-6)
                else:
                    eff = s/st['er']
                    st['er'] = (1-ALPHA)*st['er'] + ALPHA*s
                win = 1.0 if eff > 1.0 else 0.0
                a = 2.0/(EWMA_SPAN+1)
                st['ew'] = a*win + (1-a)*st['ew']
                inst = eff*st['ew']
                st['pw'] = np.exp(-LAMBDA)*st['pw'] + (1-np.exp(-LAMBDA))*inst
                all_powers[b].append(st['pw'])

        final_p = np.array([all_powers[b][-1] for b in BENCHMARKS])
        final_s = np.array([step_dom[steps[-1]][b] for b in BENCHMARKS])

        # Softmax proportions
        ex = np.exp(final_p - final_p.max())
        props = ex / ex.sum()

        # Pearson r with bootstrap (n=5 — wide CIs guaranteed)
        r, _ = pointbiserialr([1]*5, final_p) if False else (np.corrcoef(final_p, final_s)[0,1], None)
        r = float(np.corrcoef(final_p, final_s)[0,1])
        boot_r = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, 5, 5)
            try:
                if np.std(final_p[idx]) > 1e-8 and np.std(final_s[idx]) > 1e-8:
                    boot_r.append(np.corrcoef(final_p[idx], final_s[idx])[0,1])
            except Exception:
                pass
        if boot_r:
            r_lo, r_hi = np.percentile(boot_r, [2.5, 97.5])
        else:
            r_lo, r_hi = float('nan'), float('nan')

        print(f"\n  {model}")
        print(f"    {'Domain':<22} {'P_i':>7} {'Score':>7} {'Prop':>7} {'Δ unif':>9}")
        print(f"    {'-'*60}")
        for i, b in enumerate(BENCHMARKS):
            print(f"    {DOMAIN_NAMES[b]:<22} {final_p[i]:>7.4f} "
                  f"{final_s[i]:>7.4f} {props[i]:>7.3f} {props[i]-0.2:>+9.3f}")
        print(f"    Pearson r(P_i, score) = {r:+.3f}  "
              f"95% CI [{r_lo:+.3f}, {r_hi:+.3f}]   (n=5, very wide)")

        final_props_by_model[model] = props
        final_scores_by_model[model] = final_s

    # Cross-scale consistency — the actual claim of Paper 4
    print(f"\n  CROSS-SCALE PROPORTION CONSISTENCY (Paper 4's actual claim)")
    print(f"  Spearman ρ on dynamic proportions across model scales:")
    print(f"  {'-'*60}")
    pairs = [('pythia-160m','pythia-1.4b'),
             ('pythia-1.4b','pythia-6.9b'),
             ('pythia-160m','pythia-6.9b')]
    for a, b in pairs:
        rho, p = spearmanr(final_props_by_model[a], final_props_by_model[b])
        print(f"    {a:<14} vs {b:<14}  ρ={rho:+.3f}  p={p:.3f}")
    print(f"\n  (n=5 domains. p>0.05 likely for any pair — very limited power.)")


# ── Section 6: Training health ──────────────────────────────────────────

def section6_analysis(thresholds=(0.5, 0.6, 0.7), k=10):
    print("\n" + "="*78)
    print("SECTION 6 — Training Health Monitoring (Paper 1 replication)")
    print("="*78)

    rng = np.random.default_rng(RNG_SEED)

    for model in MODELS:
        steps, scores = load_composite(model)
        n = len(scores)
        powers = compute_pm(scores)

        print(f"\n  {model}  (n={n} post-warmup ckpts)")
        print(f"    {'θ':<5} {'C(t) sum':>9} {'Saved %':>9} {'Bootstrap CI':<22}")
        print(f"    {'-'*60}")
        for th in thresholds:
            ct = 1.0 / (1.0 + np.exp(-k * (powers - th)))
            saved = 1.0 - ct.sum() / n

            # Bootstrap over checkpoints
            boot = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, n, n)
                ct_b = 1.0 / (1.0 + np.exp(-k * (powers[idx] - th)))
                boot.append(1.0 - ct_b.sum() / n)
            lo, hi = np.percentile(boot, [2.5, 97.5])

            print(f"    {th:<5.1f} {ct.sum():>9.2f} {saved*100:>8.1f}%  "
                  f"[{lo*100:.1f}, {hi*100:.1f}]%")

    print(f"\n  NOTE: Score Δ = +0.000 by construction. These are *candidate*")
    print(f"  reductions — what fraction P(t) flags below threshold. Validation")
    print(f"  requires controlled allocation experiments (Paper 1 admits this).")


if __name__ == '__main__':
    print("="*78)
    print("Pythia Verifications — Sections 4, 5, 6")
    print(f"α={ALPHA}, λ={LAMBDA}, span={EWMA_SPAN}, "
          f"bootstrap={N_BOOTSTRAP}, seed={RNG_SEED}")
    print("="*78)

    if not os.path.exists(BASE):
        print(f"ERROR: {BASE} not found")
        exit(1)

    section4_analysis()
    section5_analysis()
    section6_analysis()

    print("\nDone.")
