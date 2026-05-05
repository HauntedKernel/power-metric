"""
Paper 7 Variant Sweep — EWMA vs Holt-Style Two-Timescale P(t)
==============================================================
Tests whether structural variants of P(t) strengthen, weaken, or
leave the slow-starter recovery result unchanged.

Variants:
  ewma     — published baseline. R is single-timescale EMA of s.
  holt     — two-timescale. R fast-tracks s; R_tilde slow-tracks R;
             E(t) = s(t) / R_tilde(t-1).
  presmooth — smooth s(t) into s_tilde first, then everything as ewma
             on s_tilde. Tests "denoise input first" intuition.
  deviation — additive: E(t) = 1 + (s(t) - R(t-1)). No ratio.
             Tests whether ratio vs deviation matters.

For each variant, sweeps across:
  - threshold θ in [0.3, 0.7]
  - rung position in [ckpt 1..5]
  - eta in {2, 3, 4}
  - relevant smoothing parameters

Outputs a summary table per variant: number of slow starters recovered,
robustness across grid, false positive count.

Run from C:\\Users\\Carolina\\:
    python paper7_variants.py
"""

import os, json, re
import numpy as np
from itertools import product

RNG_SEED = 42
BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']
TARGET_MODELS = ['pythia-70m','pythia-160m','pythia-410m',
                 'pythia-1.4b','pythia-2.8b','pythia-6.9b','pythia-12b']
SLOW_STARTER_THRESHOLD = 0.05
BASE = './pythia-main/evals/pythia-v1'


# ── Data loading ────────────────────────────────────────────────────────

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
    if len(steps) < 12: return None
    return np.array([step_scores[s] for s in steps])


# ── Variant implementations ─────────────────────────────────────────────

def pm_ewma(scores, alpha=0.3, lam=0.5, span=3):
    """Published baseline. Single-timescale EWMA on R."""
    er = None; ew = 0.0; pw = 0.0; out = []
    for s in scores:
        if er is None:
            eff = 1.0; er = max(s, 1e-6)
        else:
            eff = s / er
            er = (1-alpha)*er + alpha*s
        win = 1.0 if eff > 1.0 else 0.0
        a = 2.0/(span+1)
        ew = a*win + (1-a)*ew
        inst = eff*ew
        pw = np.exp(-lam)*pw + (1-np.exp(-lam))*inst
        out.append(pw)
    return np.array(out)


def pm_holt(scores, alpha=0.3, gamma=0.1, lam=0.5, span=3):
    """
    Two-timescale Holt-style.
    R fast-tracks s.
    R_tilde slow-tracks R.
    E(t) = s(t) / R_tilde(t-1) — efficiency vs slower benchmark.
    """
    er = None; er_tilde = None
    ew = 0.0; pw = 0.0; out = []
    for s in scores:
        if er is None:
            eff = 1.0
            er = max(s, 1e-6)
            er_tilde = er
        else:
            # Compute efficiency vs slow baseline (pre-update)
            eff = s / er_tilde
            # Update fast R
            er = (1-alpha)*er + alpha*s
            # Update slow R_tilde from fast R
            er_tilde = (1-gamma)*er_tilde + gamma*er
        win = 1.0 if eff > 1.0 else 0.0
        a = 2.0/(span+1)
        ew = a*win + (1-a)*ew
        inst = eff*ew
        pw = np.exp(-lam)*pw + (1-np.exp(-lam))*inst
        out.append(pw)
    return np.array(out)


def pm_presmooth(scores, alpha=0.3, mu=0.4, lam=0.5, span=3):
    """
    Pre-smooth input s(t) into s_tilde(t), then standard EWMA pipeline
    on s_tilde. mu controls the input smoothing aggressiveness.
    """
    s_tilde_prev = None
    er = None; ew = 0.0; pw = 0.0; out = []
    for s in scores:
        # Pre-smooth the input
        if s_tilde_prev is None:
            s_tilde = s
        else:
            s_tilde = (1-mu)*s_tilde_prev + mu*s
        s_tilde_prev = s_tilde

        if er is None:
            eff = 1.0; er = max(s_tilde, 1e-6)
        else:
            eff = s_tilde / er
            er = (1-alpha)*er + alpha*s_tilde
        win = 1.0 if eff > 1.0 else 0.0
        a = 2.0/(span+1)
        ew = a*win + (1-a)*ew
        inst = eff*ew
        pw = np.exp(-lam)*pw + (1-np.exp(-lam))*inst
        out.append(pw)
    return np.array(out)


def pm_deviation(scores, alpha=0.3, lam=0.5, span=3):
    """
    Additive efficiency: E(t) = 1 + (s(t) - R(t-1)).
    Tests whether the ratio formulation specifically matters,
    or whether any signed surprise signal works.
    Uses centered comparison to win threshold.
    """
    er = None; ew = 0.0; pw = 0.0; out = []
    for s in scores:
        if er is None:
            eff = 1.0; er = s
        else:
            eff = 1.0 + (s - er)   # additive surprise centered at 1
            er = (1-alpha)*er + alpha*s
        win = 1.0 if eff > 1.0 else 0.0
        a = 2.0/(span+1)
        ew = a*win + (1-a)*ew
        # Clip eff to non-negative to keep P(t) sensible
        eff_clipped = max(eff, 0.0)
        inst = eff_clipped * ew
        pw = np.exp(-lam)*pw + (1-np.exp(-lam))*inst
        out.append(pw)
    return np.array(out)


VARIANTS = {
    'ewma':      pm_ewma,
    'holt':      pm_holt,
    'presmooth': pm_presmooth,
    'deviation': pm_deviation,
}


# ── HB + PM evaluation ──────────────────────────────────────────────────

def hyperband_eliminate(curves, rung_idx, eta=3):
    rung_scores = {n: c[rung_idx] for n, c in curves.items()}
    sorted_by = sorted(rung_scores.items(), key=lambda x: x[1])
    n_elim = int(len(curves) * (1 - 1/eta))
    return set(n for n, _ in sorted_by[:n_elim])


def evaluate(curves, powers_by_config, rung1_idx, theta, eta):
    elim_hb = hyperband_eliminate(curves, rung1_idx, eta)
    out = {}
    for n, c in curves.items():
        rung1 = c[rung1_idx]; final = c[-1]
        is_slow = (final - rung1) > SLOW_STARTER_THRESHOLD
        eliminated = n in elim_hb
        pm_at_rung1 = powers_by_config[n][rung1_idx]
        recovers = eliminated and (pm_at_rung1 > theta)
        out[n] = dict(
            rung1=rung1, final=final, is_slow=is_slow,
            eliminated=eliminated, pm_at_rung1=pm_at_rung1,
            pm_recovers=recovers,
        )
    return out


def score_grid(variant_fn, variant_kwargs, curves, grid):
    """
    Score a variant across the full grid.
    Returns aggregate stats.
    """
    powers = {n: variant_fn(c, **variant_kwargs) for n, c in curves.items()}
    n_grid = 0
    n_recover = 0
    n_false_pos = 0
    n_miss = 0
    config_recover_counts = {n: 0 for n in curves}

    for rung1, theta, eta in grid:
        out = evaluate(curves, powers, rung1, theta, eta)
        n_grid += 1
        for n, o in out.items():
            if o['pm_recovers']:
                if o['is_slow']:
                    n_recover += 1
                    config_recover_counts[n] += 1
                else:
                    n_false_pos += 1
            elif o['eliminated'] and o['is_slow']:
                n_miss += 1

    return dict(
        n_grid_points=n_grid,
        recoveries_total=n_recover,
        false_positives_total=n_false_pos,
        slow_misses_total=n_miss,
        config_recover_counts=config_recover_counts,
        powers=powers,
    )


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("="*78)
    print("Paper 7 Variant Sweep — Structural variations of P(t)")
    print("="*78)

    curves = {}
    for m in TARGET_MODELS:
        c = load_curve(m)
        if c is not None: curves[m] = c
    n_ckpts = min(len(c) for c in curves.values())
    print(f"\nLoaded {len(curves)} configs, {n_ckpts} ckpts each")

    # Build evaluation grid
    grid = list(product(
        range(1, 5),                     # rung1 in {ckpt 1..4}
        [0.3, 0.4, 0.5, 0.6, 0.7],      # theta
        [2, 3, 4],                       # eta
    ))
    print(f"Grid size: {len(grid)} (rung × theta × eta)\n")

    # Per-variant default param sweep
    variant_configs = {
        'ewma': [
            dict(alpha=0.3, lam=0.5, span=3),  # published default
        ],
        'holt': [
            dict(alpha=0.3, gamma=0.05, lam=0.5, span=3),
            dict(alpha=0.3, gamma=0.10, lam=0.5, span=3),
            dict(alpha=0.3, gamma=0.20, lam=0.5, span=3),
            dict(alpha=0.5, gamma=0.10, lam=0.5, span=3),
            dict(alpha=0.5, gamma=0.20, lam=0.5, span=3),
        ],
        'presmooth': [
            dict(alpha=0.3, mu=0.2, lam=0.5, span=3),
            dict(alpha=0.3, mu=0.4, lam=0.5, span=3),
            dict(alpha=0.3, mu=0.6, lam=0.5, span=3),
        ],
        'deviation': [
            dict(alpha=0.3, lam=0.5, span=3),
            dict(alpha=0.5, lam=0.5, span=3),
        ],
    }

    print(f"{'Variant':<12} {'Params':<35} "
          f"{'Recov':>6} {'FP':>5} {'Miss':>5} {'Configs Recovered'}")
    print("-"*78)

    all_results = {}
    for vname, cfgs in variant_configs.items():
        fn = VARIANTS[vname]
        for kwargs in cfgs:
            res = score_grid(fn, kwargs, curves, grid)
            param_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
            recovered_configs = [n.replace('pythia-','')
                                 for n, c in res['config_recover_counts'].items()
                                 if c > 0]
            recov_str = ', '.join(recovered_configs) if recovered_configs else '—'
            print(f"{vname:<12} {param_str:<35} "
                  f"{res['recoveries_total']:>6} "
                  f"{res['false_positives_total']:>5} "
                  f"{res['slow_misses_total']:>5} "
                  f"{recov_str}")
            all_results[(vname, str(kwargs))] = res

    # Headline check on each variant at the default theta=0.5, eta=3, rung1=2
    print(f"\n{'─'*78}")
    print("HEADLINE CHECK: rung1=ckpt 2, eta=3, theta=0.5 (canonical setting)")
    print(f"{'─'*78}")
    print(f"{'Variant':<12} {'Params':<35} "
          f"{'P(1.4B)@R1':>10} {'Recovered':>10}")
    print("-"*78)
    for vname, cfgs in variant_configs.items():
        fn = VARIANTS[vname]
        for kwargs in cfgs:
            powers = {n: fn(c, **kwargs) for n, c in curves.items()}
            out = evaluate(curves, powers, rung1_idx=2, theta=0.5, eta=3)
            r14 = out['pythia-1.4b']
            param_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
            print(f"{vname:<12} {param_str:<35} "
                  f"{r14['pm_at_rung1']:>10.4f} "
                  f"{str(r14['pm_recovers']):>10}")

    # Per-config P(t) trajectories at default holt params for inspection
    print(f"\n{'─'*78}")
    print("INSPECTION: P(t) trajectories at default configs")
    print(f"{'─'*78}")
    print("ewma (α=0.3, λ=0.5):")
    powers_ewma = {n: pm_ewma(c) for n, c in curves.items()}
    print("holt (α=0.3, γ=0.10, λ=0.5):")
    powers_holt = {n: pm_holt(c, alpha=0.3, gamma=0.10) for n, c in curves.items()}

    print(f"\n  {'Config':<12} {'EWMA P@R1':>10} {'Holt P@R1':>10} {'Δ':>8}")
    for n in sorted(curves.keys()):
        p_e = powers_ewma[n][2]
        p_h = powers_holt[n][2]
        print(f"  {n:<12} {p_e:>10.4f} {p_h:>10.4f} {p_h-p_e:>+8.4f}")

    print(f"\n{'─'*78}")
    print("INTERPRETATION GUIDE")
    print(f"{'─'*78}")
    print("  - Look at 'Recov' column: how many slow-starter recoveries")
    print("    across the full {rung × theta × eta} grid?")
    print("    EWMA baseline gives a reference. Higher = more robust.")
    print()
    print("  - Look at 'FP' column: false positives (non-slow-starter recovered).")
    print("    A variant that recovers more by also recovering non-slow-starters")
    print("    is not actually better — it's just looser.")
    print()
    print("  - The 'Configs Recovered' column shows WHICH configs each variant")
    print("    recovers. If a variant recovers different configs than EWMA,")
    print("    that suggests the variant is detecting different signal.")
    print()
    print("  - For Kevin: the headline check shows whether 1.4B recovery is")
    print("    consistent across variants. If yes, that's a strong robustness")
    print("    statement: the recovery is not an EWMA artifact.")


if __name__ == '__main__':
    main()
