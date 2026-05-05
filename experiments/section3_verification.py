"""
Section 3 Verification — Reasoning Chain Selection
====================================================
Adds rigor to Paper 18's r=0.955 result:
  1. Bootstrap 95% CIs on Pearson r and classification accuracy
  2. 70/30 train/test split — threshold optimized on train, evaluated on test
     (the existing 100% accuracy at θ=0.65 is in-sample threshold tuning)
  3. Math-Shepherd cross-dataset generalization

Required downloads (run once before this script):
  python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='trl-lib/prm800k', repo_type='dataset', \
    local_dir='./prm800k_parquet')"
"""

import os, glob, re
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

ALPHA       = 0.3
LAMBDA      = 0.5
EWMA_SPAN   = 3
RNG_SEED    = 42
N_BOOTSTRAP = 5000   # 5K is plenty for 30K chains

PRM_PARQUET_DIR = './prm800k_parquet'
MS_LOCAL        = './math_shepherd_local'   # peiyi9979 version


def compute_pm(labels):
    er=None; ew=0.0; pw=0.0; powers=[]
    for correct in labels:
        s = 1.0 if correct else 0.0
        if er is None:
            eff = 1.0
            er  = max(s, 0.5)
        else:
            eff = s / er if er > 1e-6 else (1.0 if s > 0 else 0.0)
            er  = (1 - ALPHA) * er + ALPHA * s
        win  = 1.0 if eff >= 1.0 else 0.0
        a    = 2.0 / (EWMA_SPAN + 1)
        ew   = a * win + (1 - a) * ew
        inst = eff * ew
        pw   = np.exp(-LAMBDA) * pw + (1 - np.exp(-LAMBDA)) * inst
        powers.append(pw)
    return np.array(powers)


def chain_features(labels):
    powers = compute_pm(labels)
    n = len(labels)
    return dict(
        pm_final     = powers[-1],
        running_acc  = sum(labels) / n,
        last5_acc    = sum(labels[-5:]) / 5 if n >= 5 else sum(labels) / n,
        all_correct  = all(labels),
        n_steps      = n,
    )


def best_threshold(scores, labels):
    grid = np.arange(0.05, 1.0, 0.01)
    best, best_t = 0.0, 0.0
    for t in grid:
        acc = ((scores > t) == labels).mean()
        if acc > best:
            best, best_t = acc, t
    return best, best_t


def bootstrap_corr(x, y, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            r, _ = pointbiserialr(y[idx], x[idx])
            rs.append(r)
        except Exception:
            continue
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def bootstrap_acc(scores, labels, threshold, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(scores)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(((scores[idx] > threshold) == labels[idx]).mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def load_prm800k(path_dir):
    """Load all parquet files in directory."""
    files = glob.glob(os.path.join(path_dir, '**/*.parquet'), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files in {path_dir}")
    print(f"  Found {len(files)} parquet file(s)")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} raw rows  |  columns: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        if 'labels' not in row or row['labels'] is None:
            continue
        labels = list(row['labels'])
        if len(labels) < 5:
            continue
        # cast to bool
        labels = [bool(l) for l in labels]
        rows.append(chain_features(labels))
    return pd.DataFrame(rows)


def parse_math_shepherd_label(label_text):
    """Extract step-level correctness from peiyi9979 format.
    Each step ends with '+' (correct) or '-' (incorrect)."""
    if not isinstance(label_text, str):
        return None
    # Find all step labels marked at end of step lines
    # peiyi9979 uses 'ки' as step delimiter, '+' or '-' as label
    matches = re.findall(r'\bки\s+([+\-])', label_text)
    if not matches:
        # Fallback: any '+' or '-' at line ends
        matches = re.findall(r'([+\-])\s*(?:\n|$)', label_text)
    if not matches:
        return None
    return [m == '+' for m in matches]


def load_math_shepherd(path):
    from datasets import load_from_disk
    ds = load_from_disk(path)
    print(f"  Splits: {list(ds.keys())}")
    split = list(ds.keys())[0]
    print(f"  Columns: {ds[split].column_names}")
    # Print first row to confirm format
    first = ds[split][0]
    sample_label = first.get('label', '')
    print(f"  First row 'label' preview: {repr(sample_label[:200])}...")

    rows = []
    fail = 0
    for row in ds[split]:
        labels = parse_math_shepherd_label(row.get('label', ''))
        if labels is None or len(labels) < 5:
            fail += 1
            continue
        rows.append(chain_features(labels))
    print(f"  Parsed {len(rows):,} chains  |  Skipped {fail:,}")
    return pd.DataFrame(rows)


def analyze(name, df):
    print(f"\n{'='*78}")
    print(f"{name}  —  {len(df):,} chains  ({df['all_correct'].sum():,} correct)")
    print(f"{'='*78}")

    pm    = df['pm_final'].values
    racc  = df['running_acc'].values
    l5    = df['last5_acc'].values
    ac    = df['all_correct'].values

    print(f"\n  IN-SAMPLE (matching Paper 18 methodology)")
    print(f"  {'Signal':<14} {'r':>9} {'95% CI':<22} {'Acc':>7} {'CI':<18} {'θ*':>5}")
    print(f"  {'-'*78}")
    for nm, scores in [('P(t) final', pm), ('Running acc', racc), ('Last-5 acc', l5)]:
        r, _ = pointbiserialr(ac, scores)
        rlo, rhi = bootstrap_corr(scores, ac)
        acc, theta = best_threshold(scores, ac)
        alo, ahi = bootstrap_acc(scores, ac, theta)
        print(f"  {nm:<14} {r:>+8.4f} [{rlo:+.3f},{rhi:+.3f}]  "
              f"{acc*100:>5.1f}%  [{alo*100:.1f},{ahi*100:.1f}]   {theta:.2f}")

    # 70/30 split
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * 0.7)
    train = df.iloc[idx[:n_train]]
    test  = df.iloc[idx[n_train:]]

    print(f"\n  OUT-OF-SAMPLE 70/30 SPLIT  ({len(train):,} train, {len(test):,} test)")
    print(f"  {'Signal':<14} {'Train θ':>8} {'Train Acc':>11} "
          f"{'Test Acc':>11} {'Test 95% CI':<18}")
    print(f"  {'-'*78}")
    for col, nm in [('pm_final','P(t) final'),
                    ('running_acc','Running acc'),
                    ('last5_acc','Last-5 acc')]:
        s_tr, l_tr = train[col].values, train['all_correct'].values
        s_te, l_te = test[col].values,  test['all_correct'].values
        train_acc, theta = best_threshold(s_tr, l_tr)
        test_acc = ((s_te > theta) == l_te).mean()
        tlo, thi = bootstrap_acc(s_te, l_te, theta)
        print(f"  {nm:<14} {theta:>8.2f} {train_acc*100:>10.1f}%  "
              f"{test_acc*100:>10.1f}%  [{tlo*100:.1f},{thi*100:.1f}]")


def cross_dataset(prm_df, ms_df):
    """Train threshold on PRM800K, evaluate on Math-Shepherd."""
    print(f"\n{'='*78}")
    print(f"CROSS-DATASET: PRM800K → Math-Shepherd")
    print(f"{'='*78}")
    print(f"  {'Signal':<14} {'PRM θ*':>8} {'MS Acc':>8} {'95% CI':<20}")
    print(f"  {'-'*78}")
    for col, nm in [('pm_final','P(t) final'),
                    ('running_acc','Running acc'),
                    ('last5_acc','Last-5 acc')]:
        s_prm, l_prm = prm_df[col].values, prm_df['all_correct'].values
        s_ms,  l_ms  = ms_df[col].values,  ms_df['all_correct'].values
        _, theta = best_threshold(s_prm, l_prm)
        ms_acc = ((s_ms > theta) == l_ms).mean()
        lo, hi = bootstrap_acc(s_ms, l_ms, theta)
        print(f"  {nm:<14} {theta:>8.2f} {ms_acc*100:>7.1f}%  [{lo*100:.1f},{hi*100:.1f}]")


if __name__ == '__main__':
    print("="*78)
    print("Section 3 Verification — PRM800K + Math-Shepherd")
    print(f"α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}, "
          f"bootstrap={N_BOOTSTRAP}, seed={RNG_SEED}")
    print("="*78)

    prm_df = None
    ms_df  = None

    print("\nLoading PRM800K...")
    if os.path.exists(PRM_PARQUET_DIR):
        try:
            prm_df = load_prm800k(PRM_PARQUET_DIR)
            analyze('PRM800K', prm_df)
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"  Not found: {PRM_PARQUET_DIR}")
        print(f"  Run: python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download(repo_id='trl-lib/prm800k', repo_type='dataset', "
              f"local_dir='./prm800k_parquet')\"")

    print("\nLoading Math-Shepherd...")
    if os.path.exists(MS_LOCAL):
        try:
            ms_df = load_math_shepherd(MS_LOCAL)
            if len(ms_df) > 0:
                analyze('Math-Shepherd', ms_df)
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  If parsing fails, paste the 'First row label preview' "
                  f"above and we'll fix the parser.")
    else:
        print(f"  Not found: {MS_LOCAL}")

    if prm_df is not None and ms_df is not None and len(ms_df) > 0:
        cross_dataset(prm_df, ms_df)

    print("\nDone.")
