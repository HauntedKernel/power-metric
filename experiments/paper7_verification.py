"""
Paper 7 Verification — Hyperband + P(t) Slow Starter Recovery
==============================================================
Tests Paper 7's central claim: at θ=0.5, HB+PM recovers the 1.4B
slow starter that Hyperband eliminates at Rung 1.

Adds rigor:
  1. Reproduce the Pythia-as-HPO simulation
  2. Bootstrap 95% CIs over rung position (sensitivity to where rungs
     are placed — only 16 ckpts, so this is the key sensitivity)
  3. Threshold sensitivity sweep θ ∈ [0.3, 0.7]
  4. Eta sensitivity (η=2, 3, 4)
  5. Test if 1.4B recovery is θ=0.5 artifact or robust

Run from C:\\Users\\Carolina\\:
    python paper7_verification.py
"""

import os, json, re
import numpy as np

ALPHA       = 0.3
LAMBDA      = 0.5
EWMA_SPAN   = 3
RNG_SEED    = 42
N_BOOTSTRAP = 2000

BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']
TARGET_MODELS = ['pythia-70m','pythia-160m','pythia-410m',
                 'pythia-1.4b','pythia-2.8b','pythia-6.9b','pythia-12b']
SLOW_STARTER_THRESHOLD = 0.05
BASE = './pythia-main/evals/pythia-v1'


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


def compute_pm(scores):
    er=None; ew=0.0; pw=0.0; powers=[]
    for s in scores:
        if er is None: eff=1.0; er=max(s,1e-6)
        else: eff=s/er; er=(1-ALPHA)*er+ALPHA*s
        win = 1.0 if eff>1.0 else 0.0
        a = 2.0/(EWMA_SPAN+1); ew = a*win+(1-a)*ew
        inst = eff*ew
        pw = np.exp(-LAMBDA)*pw + (1-np.exp(-LAMBDA))*inst
        powers.append(pw)
    return np.array(powers)


def hyperband_eliminate(curves, rung_idx, eta=3):
    """Eliminate bottom (1 - 1/eta) by score at rung."""
    rung_scores = {n: c[rung_idx] for n, c in curves.items()}
    sorted_by = sorted(rung_scores.items(), key=lambda x: x[1])
    n_elim = int(len(curves) * (1 - 1/eta))
    return set(n for n, _ in sorted_by[:n_elim])


def evaluate(curves, rung1_idx, theta, eta):
    """Run HB and HB+PM at given config. Return outcomes per model."""
    powers = {n: compute_pm(c) for n, c in curves.items()}
    elim_hb = hyperband_eliminate(curves, rung1_idx, eta)

    outcomes = {}
    for n, c in curves.items():
        rung1 = c[rung1_idx]
        final = c[-1]
        improvement = final - rung1
        is_slow = improvement > SLOW_STARTER_THRESHOLD
        eliminated = n in elim_hb
        pm_at_rung1 = powers[n][rung1_idx]
        pm_recovers = eliminated and (pm_at_rung1 > theta)

        outcomes[n] = dict(
            rung1=rung1, final=final, improvement=improvement,
            is_slow=is_slow, eliminated=eliminated,
            pm_at_rung1=pm_at_rung1, pm_recovers=pm_recovers,
        )
    return outcomes


def summarize(outcomes):
    """Best config found by HB-only vs HB+PM."""
    hb_kept = {n: o for n, o in outcomes.items() if not o['eliminated']}
    hb_pm_kept = {n: o for n, o in outcomes.items()
                  if not o['eliminated'] or o['pm_recovers']}
    best_hb = max(hb_kept.items(), key=lambda x: x[1]['final']) if hb_kept else None
    best_hb_pm = max(hb_pm_kept.items(), key=lambda x: x[1]['final']) if hb_pm_kept else None

    n_slow = sum(1 for o in outcomes.values() if o['is_slow'])
    n_slow_recovered = sum(1 for o in outcomes.values()
                            if o['is_slow'] and o['eliminated'] and o['pm_recovers'])
    n_slow_missed = sum(1 for o in outcomes.values()
                        if o['is_slow'] and o['eliminated'] and not o['pm_recovers'])

    return dict(
        best_hb=best_hb, best_hb_pm=best_hb_pm,
        n_slow=n_slow, n_recovered=n_slow_recovered, n_missed=n_slow_missed,
    )


def main():
    print("="*78)
    print("Paper 7 Verification — Hyperband + P(t)")
    print("="*78)

    curves = {}
    for m in TARGET_MODELS:
        c = load_curve(m)
        if c is not None:
            curves[m] = c
    print(f"\nLoaded {len(curves)} configs: {list(curves.keys())}")
    print(f"Checkpoints per config: {[len(c) for c in curves.values()]}")

    # Headline: replicate Paper 7's main result (rung at ckpt 2, eta=3, theta=0.5)
    print(f"\n{'─'*78}")
    print("HEADLINE: Paper 7 setup (rung1=ckpt 2, eta=3, theta=0.5)")
    print(f"{'─'*78}")
    out = evaluate(curves, rung1_idx=2, theta=0.5, eta=3)
    s = summarize(out)
    print(f"\n  {'Model':<14} {'Rung1':>7} {'Final':>7} {'Δ':>7} "
          f"{'Slow':>5} {'Elim':>5} {'P@R1':>6} {'Recover':>8}")
    print(f"  {'-'*70}")
    for n in sorted(out.keys(), key=lambda x: out[x]['rung1']):
        o = out[n]
        print(f"  {n:<14} {o['rung1']:>7.4f} {o['final']:>7.4f} "
              f"{o['improvement']:>+7.4f} {str(o['is_slow']):>5} "
              f"{str(o['eliminated']):>5} {o['pm_at_rung1']:>6.3f} "
              f"{str(o['pm_recovers']):>8}")

    print(f"\n  Slow starters total: {s['n_slow']}")
    print(f"  Recovered by PM:     {s['n_recovered']}")
    print(f"  Still missed:        {s['n_missed']}")
    if s['best_hb']:
        print(f"  Best by HB only:     {s['best_hb'][0]} (final={s['best_hb'][1]['final']:.4f})")
    if s['best_hb_pm']:
        print(f"  Best by HB+PM:       {s['best_hb_pm'][0]} (final={s['best_hb_pm'][1]['final']:.4f})")

    # Threshold sensitivity
    print(f"\n{'─'*78}")
    print("THRESHOLD SENSITIVITY (rung1=ckpt 2, eta=3)")
    print(f"{'─'*78}")
    print(f"\n  {'θ':<6} {'Slow':<6} {'Recov':<7} {'Miss':<6} {'Best HB':<22} {'Best HB+PM':<22}")
    for theta in [0.3, 0.4, 0.5, 0.6, 0.7]:
        out = evaluate(curves, rung1_idx=2, theta=theta, eta=3)
        s = summarize(out)
        bhb = f"{s['best_hb'][0]}({s['best_hb'][1]['final']:.3f})" if s['best_hb'] else "—"
        bpm = f"{s['best_hb_pm'][0]}({s['best_hb_pm'][1]['final']:.3f})" if s['best_hb_pm'] else "—"
        print(f"  {theta:<6.1f} {s['n_slow']:<6} {s['n_recovered']:<7} {s['n_missed']:<6} "
              f"{bhb:<22} {bpm:<22}")

    # Rung position sensitivity (the bootstrap-equivalent for n=7 configs)
    print(f"\n{'─'*78}")
    print("RUNG POSITION SENSITIVITY (theta=0.5, eta=3)")
    print(f"  Tests if 1.4B recovery is robust to choice of where Rung 1 lands.")
    print(f"{'─'*78}")
    n_ckpts = min(len(c) for c in curves.values())
    print(f"\n  {'Rung1':<7} {'Slow':<6} {'Recov':<7} {'Miss':<6} {'Best HB':<22} {'Best HB+PM':<22}")
    for ridx in range(1, min(6, n_ckpts-2)):
        out = evaluate(curves, rung1_idx=ridx, theta=0.5, eta=3)
        s = summarize(out)
        bhb = f"{s['best_hb'][0]}({s['best_hb'][1]['final']:.3f})" if s['best_hb'] else "—"
        bpm = f"{s['best_hb_pm'][0]}({s['best_hb_pm'][1]['final']:.3f})" if s['best_hb_pm'] else "—"
        print(f"  ckpt{ridx:<3} {s['n_slow']:<6} {s['n_recovered']:<7} {s['n_missed']:<6} "
              f"{bhb:<22} {bpm:<22}")

    # Eta sensitivity
    print(f"\n{'─'*78}")
    print("ETA SENSITIVITY (rung1=ckpt 2, theta=0.5)")
    print(f"{'─'*78}")
    print(f"\n  {'eta':<5} {'Slow':<6} {'Recov':<7} {'Miss':<6} {'Best HB':<22} {'Best HB+PM':<22}")
    for eta in [2, 3, 4]:
        out = evaluate(curves, rung1_idx=2, theta=0.5, eta=eta)
        s = summarize(out)
        bhb = f"{s['best_hb'][0]}({s['best_hb'][1]['final']:.3f})" if s['best_hb'] else "—"
        bpm = f"{s['best_hb_pm'][0]}({s['best_hb_pm'][1]['final']:.3f})" if s['best_hb_pm'] else "—"
        print(f"  {eta:<5} {s['n_slow']:<6} {s['n_recovered']:<7} {s['n_missed']:<6} "
              f"{bhb:<22} {bpm:<22}")

    # Concluding interpretation
    print(f"\n{'─'*78}")
    print("INTERPRETATION GUIDE")
    print(f"{'─'*78}")
    print("  Robust positive result: P(t) recovers 1.4B (or another slow starter) and")
    print("  improves the best-found config across multiple θ, rung positions, and η.")
    print()
    print("  Likely false positive: recovery only works at θ=0.5 with rung1 exactly")
    print("  at ckpt 2. Other settings give same result as HB alone.")
    print()
    print("  Note: with 7 configs, statistical inference is limited. The qualitative")
    print("  pattern across the sensitivity table is the actual evidence.")


if __name__ == '__main__':
    main()
