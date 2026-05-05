"""
Section 4 — 160M Bidirectional Check
=====================================
Tests whether P(t) flags the 160M deduped trajectory as LESS healthy than
the 160M standard trajectory (Paper 16 shows -0.98% regression at 160M).

If yes: P(t) has a real positive detection result at 160M. The story
becomes 'P(t) detects when data quality interventions hurt small-scale
training health.' That's narrower than the original Section 4 pitch but
genuinely defensible.

If no: P(t) tracks both trajectories as similar/healthy. Section 4 doesn't
have a punchline at any scale.

Run from C:\\Users\\Carolina\\:
    python section4_160m_check.py
"""

import json, os, re
import numpy as np

ALPHA     = 0.3
LAMBDA    = 0.5
EWMA_SPAN = 3

BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']

PAIRS = [
    ('pythia-160m',  'pythia-160m-deduped',  '160M'),
    ('pythia-1.4b',  'pythia-1.4b-deduped',  '1.4B'),
    ('pythia-6.9b',  'pythia-6.9b-deduped',  '6.9B'),
    ('pythia-12b',   'pythia-12b-deduped',   '12B'),
]

BASE = './pythia-main/evals/pythia-v1'


def load_per_benchmark(model):
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
            step_scores[step] = sc
    steps = sorted(step_scores.keys())
    return steps, np.array([step_scores[s] for s in steps])


def compute_pm(scores):
    er=None; ew=0.0; pw=0.0; powers=[]
    for s in scores:
        if er is None:
            eff=1.0; er=max(s,1e-6)
        else:
            eff=s/er
            er=(1-ALPHA)*er+ALPHA*s
        win=1.0 if eff>1.0 else 0.0
        a=2.0/(EWMA_SPAN+1)
        ew=a*win+(1-a)*ew
        inst=eff*ew
        pw=np.exp(-LAMBDA)*pw+(1-np.exp(-LAMBDA))*inst
        powers.append(pw)
    return np.array(powers)


def first_persistent(a, b, k=3):
    """First i where a[i]>b[i] for k consecutive ckpts."""
    streak = 0
    for i in range(len(a)):
        if a[i] > b[i]:
            streak += 1
            if streak >= k:
                return i - k + 1
        else:
            streak = 0
    return -1


def analyze(label, std_model, ded_model, verbose_pm=False):
    steps_s, std_pb = load_per_benchmark(std_model)
    steps_d, ded_pb = load_per_benchmark(ded_model)

    common = sorted(set(steps_s) & set(steps_d))
    s_idx = [steps_s.index(s) for s in common]
    d_idx = [steps_d.index(s) for s in common]
    std_pb = std_pb[s_idx]
    ded_pb = ded_pb[d_idx]

    std_avg = std_pb.mean(axis=1)
    ded_avg = ded_pb.mean(axis=1)

    # Raw direction counts
    raw_ded_wins = int(np.sum(ded_avg > std_avg))
    raw_std_wins = int(np.sum(std_avg > ded_avg))

    # P(t)
    pm_std = compute_pm(std_avg)
    pm_ded = compute_pm(ded_avg)

    # Bidirectional persistence on P(t)
    pm_ded_persist = first_persistent(pm_ded, pm_std, k=3)
    pm_std_persist = first_persistent(pm_std, pm_ded, k=3)

    # P(t) sign counts
    pm_ded_wins = int(np.sum(pm_ded > pm_std))
    pm_std_wins = int(np.sum(pm_std > pm_ded))

    final_raw = ded_avg[-1] - std_avg[-1]
    final_pm  = pm_ded[-1] - pm_std[-1]

    print(f"\n--- {label}  ({len(common)} ckpts) ---")
    print(f"  Final raw gap (ded-std):  {final_raw:+.4f}  "
          f"({final_raw/std_avg[-1]*100:+.2f}%)")
    print(f"  Final P(t) gap (ded-std): {final_pm:+.4f}")
    print(f"  Raw counts:  ded > std at {raw_ded_wins}/{len(common)}  "
          f"|  std > ded at {raw_std_wins}/{len(common)}")
    print(f"  P(t) counts: ded > std at {pm_ded_wins}/{len(common)}  "
          f"|  std > ded at {pm_std_wins}/{len(common)}")
    print(f"  First 3-in-a-row P(ded) > P(std): "
          f"{pm_ded_persist if pm_ded_persist>=0 else 'never'}")
    print(f"  First 3-in-a-row P(std) > P(ded): "
          f"{pm_std_persist if pm_std_persist>=0 else 'never'}")

    if verbose_pm:
        print(f"\n  P(t) trajectories (160M):")
        print(f"  {'ckpt':<6} {'P_std':<8} {'P_ded':<8} {'Δ':<10}")
        for i, (ps, pd) in enumerate(zip(pm_std, pm_ded)):
            mark = '←' if abs(ps - pd) > 0.001 else ''
            print(f"  {i:<6} {ps:<8.4f} {pd:<8.4f} {pd-ps:+.4f}    {mark}")

    return dict(
        label=label,
        final_raw=final_raw, final_pm=final_pm,
        raw_ded_wins=raw_ded_wins, raw_std_wins=raw_std_wins,
        pm_ded_wins=pm_ded_wins, pm_std_wins=pm_std_wins,
        pm_ded_persist=pm_ded_persist, pm_std_persist=pm_std_persist,
    )


if __name__ == '__main__':
    print("="*72)
    print("Section 4 — 160M Bidirectional P(t) Check")
    print("="*72)

    results = []
    for std_m, ded_m, label in PAIRS:
        r = analyze(label, std_m, ded_m, verbose_pm=(label == '160M'))
        results.append(r)

    print("\n" + "="*72)
    print("THE QUESTION FOR 160M")
    print("="*72)
    r160 = next(r for r in results if r['label'] == '160M')
    print(f"  P(t) standard wins at {r160['pm_std_wins']}/16 checkpoints")
    print(f"  P(t) persistence (std > ded for 3+): "
          f"{r160['pm_std_persist'] if r160['pm_std_persist']>=0 else 'never'}")

    if r160['pm_std_persist'] >= 0 and r160['pm_std_persist'] <= 5:
        print(f"\n  POSITIVE: P(t) flags deduped 160M as less healthy "
              f"starting at ckpt {r160['pm_std_persist']}.")
        print(f"  Section 4 has a real claim at 160M.")
    elif r160['pm_std_wins'] > 10:
        print(f"\n  WEAK POSITIVE: P(t) leans toward standard at 160M but no "
              f"clean persistence.")
        print(f"  Could support a softer Section 4 claim with caveats.")
    else:
        print(f"\n  NEGATIVE: P(t) does not separate the trajectories at 160M.")
        print(f"  Section 4 has no detection story to tell. Recommend cut.")
