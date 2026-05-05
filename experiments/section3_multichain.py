"""
Section 3 v2 — Multi-Chain-Per-Problem Selection
=================================================
The actual test-time scaling task: given N candidate chains for the same
problem, select the best one. PRM800K's structure has multiple chains
per problem (the 'completions' field). We test if P(t) selects the best
chain better than running accuracy or last-5.

Also fixes Math-Shepherd loading from raw parquet (not HF dataset format).

If your previous Section 3 verification confused you with Last-5 = 100%,
this is the harder, more realistic test. Here Last-5 cannot trivially
win — the choice is among multiple correct/incorrect chains for ONE
problem, not classifying isolated chains.

Run:
    python section3_multichain.py
"""

import os, glob, re
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

ALPHA       = 0.3
LAMBDA      = 0.5
EWMA_SPAN   = 3
RNG_SEED    = 42
N_BOOTSTRAP = 2000

PRM_PARQUET_DIR = './prm800k_parquet'
MS_DIR = './math_shepherd_local'   # raw parquet from snapshot_download


def compute_pm(labels):
    er=None; ew=0.0; pw=0.0; powers=[]
    for correct in labels:
        s = 1.0 if correct else 0.0
        if er is None: eff=1.0; er=max(s, 0.5)
        else:
            eff = s/er if er>1e-6 else (1.0 if s>0 else 0.0)
            er  = (1-ALPHA)*er + ALPHA*s
        win = 1.0 if eff>=1.0 else 0.0
        a = 2.0/(EWMA_SPAN+1); ew = a*win + (1-a)*ew
        inst = eff*ew
        pw = np.exp(-LAMBDA)*pw + (1-np.exp(-LAMBDA))*inst
        powers.append(pw)
    return np.array(powers)


def chain_score(labels):
    if len(labels) < 1: return None
    powers = compute_pm(labels)
    n = len(labels)
    return dict(
        pm_final = float(powers[-1]),
        running_acc = sum(labels)/n,
        last5_acc = sum(labels[-5:])/min(5,n),
        all_correct = bool(all(labels)),
        n_steps = n,
    )


def explore_prm800k_structure():
    """Show what PRM800K columns look like — 'completions' should be a list."""
    files = sorted(glob.glob(os.path.join(PRM_PARQUET_DIR, '**/*.parquet'), recursive=True))
    if not files: return None
    df = pd.read_parquet(files[0])
    print(f"  PRM800K columns: {list(df.columns)}")
    print(f"  First row:")
    first = df.iloc[0]
    for col in df.columns:
        val = first[col]
        s = repr(val)[:200]
        print(f"    {col}: {s}")
    return df


def explore_ms_structure():
    """Inspect raw Math-Shepherd parquet."""
    files = sorted(glob.glob(os.path.join(MS_DIR, '**/*.parquet'), recursive=True))
    if not files:
        print(f"  No parquet files found in {MS_DIR}")
        # Try other formats
        all_files = []
        for ext in ['*.json', '*.jsonl', '*.arrow', '*.csv']:
            all_files.extend(glob.glob(os.path.join(MS_DIR, '**', ext), recursive=True))
        if all_files:
            print(f"  Other files found:")
            for f in all_files[:10]:
                print(f"    {f}")
        return None
    print(f"  Math-Shepherd parquet files: {len(files)}")
    df = pd.read_parquet(files[0])
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"  First row preview:")
    first = df.iloc[0]
    for col in df.columns:
        val = first[col]
        s = repr(val)[:300]
        print(f"    {col}: {s}")
    return df


def parse_ms_label(label_text):
    """Extract step labels from Math-Shepherd 'label' field."""
    if not isinstance(label_text, str): return None
    matches = re.findall(r'\bки\s+([+\-])', label_text)
    if not matches:
        matches = re.findall(r'([+\-])\s*(?:\n|$)', label_text)
    if not matches: return None
    return [m == '+' for m in matches]


def load_prm800k_grouped(df):
    """
    PRM800K: each row has 'prompt', 'completions' (list of chains),
    and 'labels' (list of label sequences for each chain).
    Group chains by problem.
    """
    problems = {}  # prompt -> list of chains
    for _, row in df.iterrows():
        prompt = row['prompt']
        if 'completions' in row and 'labels' in row:
            comps = row['completions']
            labs = row['labels']
            # Format depends — could be list of lists or single chain
            if isinstance(labs, (list, np.ndarray)):
                # Check if it's a list of chains or a single chain
                if len(labs) > 0 and isinstance(labs[0], (list, np.ndarray, bool, np.bool_)):
                    if isinstance(labs[0], (list, np.ndarray)):
                        # List of chains
                        for chain_labels in labs:
                            cs = chain_score([bool(l) for l in chain_labels])
                            if cs and cs['n_steps'] >= 5:
                                problems.setdefault(prompt, []).append(cs)
                    else:
                        # Single chain
                        cs = chain_score([bool(l) for l in labs])
                        if cs and cs['n_steps'] >= 5:
                            problems.setdefault(prompt, []).append(cs)
    return problems


def selection_accuracy(problems, signal_col):
    """For each problem with ≥2 chains, does argmax(signal) pick a correct chain?"""
    correct_picks = 0
    valid = 0
    has_correct = 0  # problems where at least one chain is correct
    for prompt, chains in problems.items():
        if len(chains) < 2: continue
        if not any(c['all_correct'] for c in chains): continue
        valid += 1
        has_correct += 1
        scores = [c[signal_col] for c in chains]
        picked = chains[int(np.argmax(scores))]
        if picked['all_correct']:
            correct_picks += 1
    if valid == 0: return None, 0
    return correct_picks / valid, valid


def bootstrap_selection(problems, signal_col, n_boot=N_BOOTSTRAP):
    """Bootstrap CI on selection accuracy."""
    rng = np.random.default_rng(RNG_SEED)
    multi = [(p, c) for p, c in problems.items()
             if len(c) >= 2 and any(ch['all_correct'] for ch in c)]
    if not multi: return (float('nan'), float('nan'))
    n = len(multi)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [multi[i] for i in idx]
        c = 0; v = 0
        for p, chains in sample:
            v += 1
            scores = [ch[signal_col] for ch in chains]
            if chains[int(np.argmax(scores))]['all_correct']:
                c += 1
        accs.append(c/v)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    print("="*78)
    print("Section 3 v2 — Multi-Chain Selection")
    print("="*78)

    print("\n[A] PRM800K structure inspection")
    prm_df = explore_prm800k_structure()

    print("\n[B] Math-Shepherd structure inspection")
    ms_df = explore_ms_structure()

    if prm_df is None:
        print("\n  PRM800K not loaded; skipping multichain analysis.")
        return

    print("\n[C] PRM800K multi-chain selection")
    problems = load_prm800k_grouped(prm_df)
    counts = [len(v) for v in problems.values()]
    multi_count = sum(1 for c in counts if c >= 2)
    has_correct = sum(1 for chains in problems.values()
                      if len(chains) >= 2 and any(ch['all_correct'] for ch in chains))
    print(f"  Problems total: {len(problems):,}")
    print(f"  Problems w/ ≥2 chains: {multi_count:,}")
    print(f"  Problems w/ ≥2 chains AND ≥1 correct: {has_correct:,}")
    print(f"  Mean chains/problem: {np.mean(counts):.2f}  Median: {np.median(counts):.0f}")

    if has_correct == 0:
        print("\n  No multi-chain problems with a correct answer.")
        print("  This dataset format may not support best-of-N selection.")
        print("  PRM800K's structure may be one-chain-per-prompt — check schema above.")
        return

    print(f"\n  TEST: Best-of-N selection accuracy ({has_correct:,} eligible problems)")
    print(f"  {'Signal':<14} {'Accuracy':>10} {'95% CI':<22}")
    print(f"  {'-'*50}")
    for col, name in [('pm_final','P(t) final'),
                      ('running_acc','Running acc'),
                      ('last5_acc','Last-5 acc')]:
        acc, n = selection_accuracy(problems, col)
        if acc is None: continue
        lo, hi = bootstrap_selection(problems, col)
        print(f"  {name:<14} {acc*100:>9.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")

    print(f"\n  Random baseline: ~50% if balanced, varies by chain ratio")
    print(f"  Oracle ceiling: 100%")


if __name__ == '__main__':
    main()
