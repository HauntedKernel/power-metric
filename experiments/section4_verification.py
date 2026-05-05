"""
Section 4 Verification: Does P(t) Detect the Deduped Quality Gap
                        Earlier Than Raw Composite Scores?

For framework paper. The question is whether P(t)'s adaptive baseline
+ win rate + smoothing provides a detection advantage over just looking
at the raw average score, on the existing 16 Pythia checkpoints.

Three measures:
  1. Per-checkpoint paired t-test on the 5 per-benchmark differences.
     First ckpt with t > 1.645 (one-tailed 95%) = first 'significant' raw signal.
  2. Raw persistence: first ckpt where deduped > standard for 3 consecutive
     checkpoints (composite score).
  3. P(t) persistence: same, but on the P(t) values.

Run from C:\\Users\\Carolina\\:
    python section4_verification.py
"""

import json, os, re
import numpy as np

# Same parameters as the published series
ALPHA     = 0.3
LAMBDA    = 0.5
EWMA_SPAN = 3

BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']

PAIRS = [
    ('pythia-1.4b',  'pythia-1.4b-deduped',  '1.4B'),
    ('pythia-6.9b',  'pythia-6.9b-deduped',  '6.9B'),
    ('pythia-12b',   'pythia-12b-deduped',   '12B'),
]

BASE = './pythia-main/evals/pythia-v1'


def load_per_benchmark(model):
    """Return (steps_list, scores_array) where scores has shape (n_ckpt, n_bench)."""
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
    """P(t) with pre-update adaptive baseline. Same as series."""
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
    """First index i where a[i]>b[i] for k consecutive checkpoints (returns streak start)."""
    streak = 0
    for i in range(len(a)):
        if a[i] > b[i]:
            streak += 1
            if streak >= k:
                return i - k + 1
        else:
            streak = 0
    return -1


def per_ckpt_t_stat(ded_pb, std_pb):
    """Paired t-stat across the 5 benchmarks at each checkpoint."""
    diffs = ded_pb - std_pb            # (n_ckpt, n_bench)
    means = diffs.mean(axis=1)
    sems  = diffs.std(axis=1, ddof=1) / np.sqrt(diffs.shape[1])
    sems  = np.where(sems == 0, 1e-10, sems)
    return means / sems


def first_above(values, threshold):
    for i, v in enumerate(values):
        if v > threshold:
            return i
    return -1


def analyze(label, std_model, ded_model):
    steps_s, std_pb = load_per_benchmark(std_model)
    steps_d, ded_pb = load_per_benchmark(ded_model)

    common = sorted(set(steps_s) & set(steps_d))
    s_idx = [steps_s.index(s) for s in common]
    d_idx = [steps_d.index(s) for s in common]
    std_pb = std_pb[s_idx]
    ded_pb = ded_pb[d_idx]

    std_avg = std_pb.mean(axis=1)
    ded_avg = ded_pb.mean(axis=1)

    # Per-ckpt one-tailed t on per-benchmark diffs
    t_vals = per_ckpt_t_stat(ded_pb, std_pb)
    raw_first_t = first_above(t_vals, 1.645)

    # Persistence on composite scores
    raw_first_persist = first_persistent(ded_avg, std_avg, k=3)

    # Persistence on P(t)
    pm_std = compute_pm(std_avg)
    pm_ded = compute_pm(ded_avg)
    pm_first_persist = first_persistent(pm_ded, pm_std, k=3)

    final_raw = ded_avg[-1] - std_avg[-1]
    final_pm  = pm_ded[-1] - pm_std[-1]
    rel_raw   = final_raw / std_avg[-1] * 100
    rel_pm    = final_pm  / pm_std[-1]  * 100
    amp       = abs(rel_pm / rel_raw) if rel_raw != 0 else float('nan')

    print(f"\n--- {label}  ({len(common)} ckpts) ---")
    print(f"  Final raw gap : {final_raw:+.4f}  ({rel_raw:+.2f}%)")
    print(f"  Final P(t) gap: {final_pm:+.4f}  ({rel_pm:+.2f}%)")
    print(f"  P(t) amplification of relative gap: {amp:.1f}x")
    print(f"")
    print(f"  First ckpt where per-bench t > 1.645 (raw):     "
          f"{raw_first_t if raw_first_t>=0 else 'never'} / {len(common)-1}")
    print(f"  First ckpt with 3-in-a-row deduped > std (raw): "
          f"{raw_first_persist if raw_first_persist>=0 else 'never'} / {len(common)-1}")
    print(f"  First ckpt with 3-in-a-row deduped > std (P(t)):"
          f" {pm_first_persist if pm_first_persist>=0 else 'never'} / {len(common)-1}")

    return dict(
        label=label, n=len(common),
        rel_raw=rel_raw, rel_pm=rel_pm, amp=amp,
        raw_first_t=raw_first_t,
        raw_first_persist=raw_first_persist,
        pm_first_persist=pm_first_persist,
    )


if __name__ == '__main__':
    print("="*72)
    print("Section 4 Verification: P(t) vs raw composite for early-detection")
    print("="*72)
    results = [analyze(l, s, d) for s, d, l in PAIRS]

    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"{'Scale':<6} {'Raw Δ%':<9} {'P(t) Δ%':<10} {'Amp':<6} "
          f"{'Raw t-sig':<11} {'Raw persist':<13} {'P(t) persist':<14}")
    for r in results:
        rt = str(r['raw_first_t']) if r['raw_first_t']>=0 else 'never'
        rp = str(r['raw_first_persist']) if r['raw_first_persist']>=0 else 'never'
        pp = str(r['pm_first_persist']) if r['pm_first_persist']>=0 else 'never'
        print(f"{r['label']:<6} {r['rel_raw']:>+6.2f}%  "
              f"{r['rel_pm']:>+6.2f}%   {r['amp']:>4.1f}x "
              f"{rt:<11} {rp:<13} {pp:<14}")

    print("\nInterpretation guide:")
    print("  - If P(t) persist << Raw persist: P(t) detects the gap earlier")
    print("  - If similar: P(t) does not add detection value at 16-ckpt density")
    print("  - 'Amp' >> 1: P(t) amplifies the relative signal vs raw scores")
    print("  - 'Raw t-sig' is rigorous but weak (n=5 benchmarks per ckpt)")
