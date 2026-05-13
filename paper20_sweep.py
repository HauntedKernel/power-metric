"""
Paper 20 — Sensitivity sweep
============================
Runs the v3 mechanism across a 3x3 grid of (HARD_KILL_LEVEL, SCAFFOLD_DISCOUNT)
on both PRM800K and Math-Shepherd, holding K_SIGMA=3.0 and GRACE_STEPS=3 fixed.

Reports a sensitivity table per dataset:
  - Compute saving
  - Winner accuracy
  - Saving / oracle (efficiency vs achievable ceiling)
  - Killed-but-correct rate

Goal: show that the chosen operating point (L_kill=0.40, d=0.30) is
robust to nearby choices, NOT a single-point fluke.

Run from C:\\Users\\Carolina\\:
    python paper20_sweep.py
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product


# ── Datasets ──────────────────────────────────────────────────────────────
DATASETS = {
    'PRM800K': [
        './trl_prm800k_cache/trl-lib___prm800k/default-0a6a2b0aa74a9c22/0.0.0/train-00000-of-00001.parquet',
        './prm800k_parquet/data/train-00000-of-00001.parquet',
    ],
    'Math-Shepherd': [
        './math_shepherd_parquet/data/train-00000-of-00001.parquet',
    ],
}

# ── Shared (fixed) parameters ─────────────────────────────────────────────
LAMBDA            = 0.5
SPAN              = 3
ALPHA             = 0.5
MIN_STEPS         = 5
K_SIGMA           = 3.0
GRACE_STEPS       = 3
MIN_BASELINE_STD  = 0.005
OFFLINE_THRESHOLD = 0.5
MAX_PROBLEMS      = 500

# ── Swept parameters ──────────────────────────────────────────────────────
HARD_KILL_GRID     = [0.30, 0.40, 0.50]
SCAFFOLD_DISC_GRID = [0.25, 0.30, 0.35]


# ── P(t) variants ─────────────────────────────────────────────────────────

def pt_defensive(labels, alpha=ALPHA):
    er, ew, pw = 0.5, 0.0, 0.0
    a, decay   = 2.0 / (SPAN + 1), np.exp(-LAMBDA)
    out = []
    for correct in labels:
        s   = 1.0 if correct else 0.0
        eff = s / er if er > 1e-6 else (1.0 if s > 0 else 0.0)
        er  = (1 - alpha) * er + alpha * s
        win = 1.0 if eff >= 1.0 else 0.0
        ew  = a * win + (1 - a) * ew
        pw  = decay * pw + (1 - decay) * (eff * ew)
        out.append(pw)
    return np.array(out)


def pt_offensive_from_prior(labels, prior, alpha=ALPHA):
    prior = max(float(prior), 1e-6)
    eff_s = None
    ew, pw = 0.0, 0.0
    a, decay = 2.0 / (SPAN + 1), np.exp(-LAMBDA)
    out = []
    for correct in labels:
        s = 1.0 if correct else 0.0
        eff_raw = s / prior
        if eff_s is None:
            eff_s = eff_raw
        else:
            eff_s = (1 - alpha) * eff_s + alpha * eff_raw
        win = 1.0 if eff_s >= 1.0 else 0.0
        ew  = a * win + (1 - a) * ew
        pw  = decay * pw + (1 - decay) * (max(eff_s, 0) * ew)
        out.append(pw)
    return np.array(out)


# ── Detector ──────────────────────────────────────────────────────────────

def gap_at_step(trajectories, step):
    pt_vals = np.array([t[step] for t in trajectories])
    return pt_vals.max() - pt_vals.min()


def detect_regime_change(trajectories,
                         k=K_SIGMA,
                         grace=GRACE_STEPS,
                         min_std=MIN_BASELINE_STD):
    n_steps = min(len(t) for t in trajectories)
    if n_steps < grace + 1:
        return n_steps - 1
    gaps = np.array([gap_at_step(trajectories, t) for t in range(n_steps)])
    baseline_gaps = gaps[:grace]
    baseline_gap_mean = float(np.mean(baseline_gaps))
    baseline_gap_std  = max(float(np.std(baseline_gaps)), min_std)
    for t in range(grace, n_steps):
        z = (gaps[t] - baseline_gap_mean) / baseline_gap_std
        if z > k:
            return t
    return n_steps - 1


# ── Compute accounting (v3 prefix_steps fix) ──────────────────────────────

def compute_adaptive_branching(chains, branch_step, winner_idx,
                               trajectories,
                               hard_kill_level,
                               scaffold_discount):
    n            = len(chains)
    prefix_steps = branch_step + 1
    shared_cost  = n * prefix_steps

    winner_len    = len(chains[winner_idx][0])
    winner_remain = max(0, winner_len - prefix_steps)

    nonwinner_remain_cost = 0.0
    n_hard_killed         = 0
    n_scaffolded          = 0

    for i, (labels, _) in enumerate(chains):
        if i == winner_idx:
            continue
        remaining = max(0, len(labels) - prefix_steps)
        if remaining == 0:
            continue
        pt_at_branch = trajectories[i][branch_step]
        if pt_at_branch < hard_kill_level:
            n_hard_killed += 1
        else:
            nonwinner_remain_cost += remaining * scaffold_discount
            n_scaffolded += 1

    return shared_cost + winner_remain + nonwinner_remain_cost, n_hard_killed, n_scaffolded


# ── Per-group simulation ──────────────────────────────────────────────────

def simulate_problem_group(chains, hard_kill_level, scaffold_discount):
    if len(chains) < 2:
        return None
    trajectories = [pt_defensive(labels) for labels, _ in chains]
    branch_step  = detect_regime_change(trajectories)

    pt_at_branch = [(trajectories[i][branch_step], i)
                    for i in range(len(chains))]
    pt_at_branch.sort(reverse=True)
    winner_idx = pt_at_branch[0][1]

    _, winner_correct = chains[winner_idx]

    compute_std = sum(len(l) for l, _ in chains)
    compute_adaptive, n_killed, n_scaffolded = compute_adaptive_branching(
        chains, branch_step, winner_idx, trajectories,
        hard_kill_level, scaffold_discount,
    )

    # Killed-but-correct diagnostic
    killed_but_correct = 0
    for i, (_, correct) in enumerate(chains):
        if i == winner_idx:
            continue
        if trajectories[i][branch_step] < hard_kill_level:
            if correct:
                killed_but_correct += 1

    # Oracle
    oracle_correct = any(c for _, c in chains)

    return dict(
        winner_correct=winner_correct,
        oracle_correct=oracle_correct,
        n_killed=n_killed,
        killed_but_correct=killed_but_correct,
        compute_ratio=compute_adaptive / max(compute_std, 1),
    )


# ── Load helpers ──────────────────────────────────────────────────────────

def load_dataset(paths):
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        return None
    df = pd.read_parquet(path)
    groups = defaultdict(list)
    for _, row in df.iterrows():
        labels = list(row['labels'])
        if len(labels) < MIN_STEPS:
            continue
        prompt_key = hashlib.sha256(str(row['prompt']).encode('utf-8')).hexdigest()
        correct = all(labels)
        groups[prompt_key].append((labels, correct))
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    return multi


# ── Sweep runner ─────────────────────────────────────────────────────────

def run_sweep(name, paths):
    print(f"\n{'='*78}")
    print(f"Sensitivity sweep: {name}")
    print(f"{'='*78}")

    groups = load_dataset(paths)
    if groups is None:
        print(f"  No dataset found.")
        return None

    sorted_keys = sorted(groups.keys())[:MAX_PROBLEMS]
    selected_groups = [(k, groups[k]) for k in sorted_keys]
    print(f"  Multi-chain groups loaded: {len(groups)}")
    print(f"  Groups simulated:          {len(selected_groups)}")

    print(f"\n  Sweeping {len(HARD_KILL_GRID)}×{len(SCAFFOLD_DISC_GRID)} grid "
          f"= {len(HARD_KILL_GRID)*len(SCAFFOLD_DISC_GRID)} combinations")
    print(f"  Fixed: K_SIGMA={K_SIGMA}, GRACE={GRACE_STEPS}")

    # We can compute trajectories once per group, sweep parameters over the same trajectories.
    # But to keep code clean and parallel to the production script, re-run per combination.
    sweep_results = []

    print(f"\n  {'HARD_KILL':>9}  {'SCAFFOLD':>8}  {'Saving':>7}  {'WinAcc':>7}  "
          f"{'Oracle':>7}  {'%Oracle':>8}  {'KbutC':>7}")
    print(f"  {'-'*70}")

    for hkl in HARD_KILL_GRID:
        for sd in SCAFFOLD_DISC_GRID:
            results = []
            for _, chains in selected_groups:
                r = simulate_problem_group(chains, hkl, sd)
                if r:
                    results.append(r)

            winner_acc = np.mean([r['winner_correct'] for r in results])
            oracle_acc = np.mean([r['oracle_correct'] for r in results])
            compute_ratio = np.mean([r['compute_ratio'] for r in results])
            saving = (1 - compute_ratio) * 100

            # Killed-but-correct rate (only over kills)
            total_killed = sum(r['n_killed'] for r in results)
            total_kbc    = sum(r['killed_but_correct'] for r in results)
            kbc_rate     = (total_kbc / total_killed * 100) if total_killed else 0.0

            pct_of_oracle = (winner_acc / oracle_acc * 100) if oracle_acc else 0.0

            sweep_results.append(dict(
                hard_kill=hkl,
                scaffold_discount=sd,
                saving_pct=saving,
                winner_acc=winner_acc * 100,
                oracle_acc=oracle_acc * 100,
                pct_of_oracle=pct_of_oracle,
                kbc_rate=kbc_rate,
            ))

            print(f"  {hkl:>9.2f}  {sd:>8.2f}  {saving:>6.1f}%  "
                  f"{winner_acc*100:>6.1f}%  {oracle_acc*100:>6.1f}%  "
                  f"{pct_of_oracle:>7.1f}%  {kbc_rate:>6.1f}%")

    return sweep_results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("="*78)
    print("Paper 20 v3 — Sensitivity sweep across (HARD_KILL, SCAFFOLD_DISCOUNT)")
    print("="*78)

    all_results = {}
    for name, paths in DATASETS.items():
        sweep_results = run_sweep(name, paths)
        if sweep_results:
            all_results[name] = sweep_results

    # ── Cross-dataset comparison at the chosen operating point ────────────
    print(f"\n{'='*78}")
    print("Operating point (L_kill=0.40, d=0.30) across datasets")
    print(f"{'='*78}")
    op_point = (0.40, 0.30)
    print(f"\n  {'Dataset':<18} {'Saving':>8} {'WinAcc':>8} {'Oracle':>8} "
          f"{'%Oracle':>9} {'KbutC':>7}")
    print(f"  {'-'*65}")
    for name, results in all_results.items():
        for r in results:
            if (r['hard_kill'], r['scaffold_discount']) == op_point:
                print(f"  {name:<18} {r['saving_pct']:>7.1f}%  "
                      f"{r['winner_acc']:>6.1f}%  {r['oracle_acc']:>6.1f}%  "
                      f"{r['pct_of_oracle']:>8.1f}%  {r['kbc_rate']:>6.1f}%")

    # ── Range across the grid per dataset ─────────────────────────────────
    print(f"\n{'='*78}")
    print("Robustness: range of outcomes across the 3×3 grid")
    print(f"{'='*78}")
    print(f"\n  {'Dataset':<18} {'Metric':<22} {'Min':>8} {'Max':>8} {'Range':>8}")
    print(f"  {'-'*65}")
    for name, results in all_results.items():
        for metric_key, metric_label in [
            ('saving_pct',    'Saving'),
            ('winner_acc',    'Winner accuracy'),
            ('pct_of_oracle', '% of oracle'),
            ('kbc_rate',      'Killed-but-correct'),
        ]:
            vals = [r[metric_key] for r in results]
            mn, mx = min(vals), max(vals)
            print(f"  {name:<18} {metric_label:<22} "
                  f"{mn:>7.1f}%  {mx:>7.1f}%  {mx-mn:>7.1f}")

    print(f"""
{'='*78}
INTERPRETATION
{'='*78}

  Robustness check: if the operating point's (L_kill=0.40, d=0.30) is a
  single-point fluke, the grid range will be wide and reviewer-concerning.

  If the range is narrow (say, saving within ±3 pp across the grid),
  the operating point is robust and the chosen values are not cherry-picked.

  KEY THINGS TO CHECK:
    - Winner accuracy: should stay near oracle across the grid on PRM800K
    - Killed-but-correct: should stay near 0% across the grid on PRM800K
    - Math-Shepherd: saving range tells us tunability potential
    - % of oracle: how close the mechanism gets to the data ceiling
""")


if __name__ == '__main__':
    main()
