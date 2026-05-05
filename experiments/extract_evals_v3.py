"""
Multi-Family Eval Extractor v3 — Correct paths confirmed by scout
==================================================================

PolyPythias structure (confirmed by scout):
  {model}/step{N}/EleutherAI__{model}/results_{timestamp}.json
  - 4 JSON files per step at different timestamps
  - Smallest file (~13KB) is the metrics summary we want
  - Largest (~82KB) is instance-level detail, skip it

DataDecide structure (confirmed by scout):
  - Top level has 'data/' not 'models/'
  - Stored as HuggingFace parquet, not individual JSONs
  - Use load_dataset() not folder navigation

Usage:
    python extract_evals_v3.py --polypythia
    python extract_evals_v3.py --datadecide
    python extract_evals_v3.py --merge

Install:
    pip install datasets huggingface_hub pandas --break-system-packages

Run from C:\\Users\\Carolina\\
"""

import os, json, re, time, argparse
import numpy as np
import pandas as pd

OUT_DIR = '.'
DELAY = 1.5       # seconds between HF requests
DELAY_429 = 30.0  # backoff on rate limit
SHARED_BENCHMARKS = ['piqa', 'arc_easy', 'arc_challenge', 'winogrande']


# ── Shared utilities ──────────────────────────────────────────────────

def safe_download(repo_id, filename, repo_type='dataset',
                   local_dir='.', retries=5):
    from huggingface_hub import hf_hub_download
    for attempt in range(retries):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            time.sleep(DELAY)
            return path
        except Exception as e:
            msg = str(e)
            if '429' in msg or 'rate' in msg.lower():
                wait = DELAY_429 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif '404' in msg:
                return None
            else:
                print(f"    Error: {e}")
                return None
    return None


def score_from_json(data):
    """Extract mean accuracy across shared benchmarks from a results JSON."""
    scores = []
    r = data.get('results', data)
    for bench in SHARED_BENCHMARKS:
        if bench in r:
            bd = r[bench]
            s = bd.get('acc_norm', bd.get('acc', None))
            if s is not None:
                scores.append(float(s))
    return float(np.mean(scores)) if len(scores) >= 2 else None


# ══════════════════════════════════════════════════════════════════════
# SOURCE 1: PolyPythias
# Confirmed path: {model}/step{N}/EleutherAI__{model}/{smallest_json}
# ══════════════════════════════════════════════════════════════════════

def extract_polypythia(n_checkpoints=16, cache_dir='./polypythias-evals'):
    from huggingface_hub import list_repo_tree

    models = [
        'pythia-14m-seed1',
        'pythia-70m-seed1',
        'pythia-160m-seed1',
        'pythia-410m-seed1',
    ]

    print(f"\n[PolyPythias] {n_checkpoints} checkpoints × {len(models)} models")
    print(f"  Path: model/stepN/EleutherAI__model/smallest_results.json")

    rows = []

    for model in models:
        print(f"\n  Model: {model}")

        # List step directories
        try:
            step_entries = list(list_repo_tree(
                repo_id='EleutherAI/polypythias-evals',
                repo_type='dataset',
                path_in_repo=model,
                recursive=False,
            ))
            time.sleep(0.5)
        except Exception as e:
            print(f"    Cannot list steps: {e}")
            continue

        steps = {}
        for entry in step_entries:
            name = entry.path.split('/')[-1]
            m = re.match(r'step(\d+)$', name)
            if m:
                steps[int(m.group(1))] = entry.path

        if not steps:
            print(f"    No steps found")
            continue

        all_steps = sorted(steps.keys())
        # Skip step0 and step1 — too early for meaningful scores
        all_steps = [s for s in all_steps if s >= 1000]
        if len(all_steps) < n_checkpoints:
            print(f"    Only {len(all_steps)} valid steps")
            n_sample = len(all_steps)
        else:
            n_sample = n_checkpoints

        indices = np.linspace(0, len(all_steps)-1, n_sample, dtype=int)
        sampled = [all_steps[i] for i in indices]
        print(f"    Sampling {n_sample} steps from {all_steps[0]}...{all_steps[-1]}")

        for step in sampled:
            step_path = steps[step]
            # The subfolder is named EleutherAI__{model}
            subfolder = f"{step_path}/EleutherAI__{model}"

            # List files in subfolder
            try:
                files = list(list_repo_tree(
                    repo_id='EleutherAI/polypythias-evals',
                    repo_type='dataset',
                    path_in_repo=subfolder,
                    recursive=False,
                ))
                time.sleep(0.3)
            except Exception as e:
                print(f"    step{step}: cannot list subfolder: {e}")
                continue

            # Find smallest JSON file — that's the metrics summary
            # (large ~82KB files are instance-level detail)
            json_files = [
                (getattr(f, 'size', 999999), f.path)
                for f in files
                if f.path.endswith('.json')
            ]
            if not json_files:
                print(f"    step{step}: no JSON files found")
                continue

            json_files.sort()  # smallest first
            target_path = json_files[0][1]
            target_size = json_files[0][0]

            # Skip if still too large (instance-level)
            if target_size > 50000:
                print(f"    step{step}: all JSONs too large ({target_size}B), skipping")
                continue

            local_path = safe_download(
                repo_id='EleutherAI/polypythias-evals',
                filename=target_path,
                repo_type='dataset',
                local_dir=cache_dir,
            )

            if local_path and os.path.exists(local_path):
                try:
                    with open(local_path) as f:
                        data = json.load(f)
                    score = score_from_json(data)
                    if score is not None:
                        rows.append({
                            'family': 'polypythia',
                            'model': model,
                            'step': step,
                            'score': score,
                        })
                        print(f"    step{step}: {score:.4f} ✓  ({target_size}B)")
                    else:
                        print(f"    step{step}: no benchmark scores found in JSON")
                except Exception as e:
                    print(f"    step{step}: parse error: {e}")
            else:
                print(f"    step{step}: download failed")

    if not rows:
        print("\n  No rows extracted.")
        return None

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, 'evals_polypythia.csv')
    df.to_csv(out, index=False)
    print(f"\n  ✓ {len(df)} rows → {out}")
    _show_summary(df)
    return df


# ══════════════════════════════════════════════════════════════════════
# SOURCE 2: DataDecide
# Confirmed: stored as HuggingFace parquet in 'data/' directory.
# Use load_dataset() and filter — no folder navigation needed.
# ══════════════════════════════════════════════════════════════════════

def extract_datadecide(n_checkpoints=16, cache_dir='./datadecide-evals'):
    """
    Load DataDecide eval results via load_dataset (parquet format).
    Filter to one training recipe and select model sizes spanning a
    range comparable to Pythia's 7 configs.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Run: pip install datasets")
        return None

    print("\n[DataDecide] Loading via load_dataset (parquet format)...")
    print("  This downloads a subset of the dataset — may take a few minutes.")

    try:
        # Stream to avoid downloading everything
        ds = load_dataset(
            'allenai/DataDecide-eval-results',
            split='train',
            streaming=True,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"  load_dataset failed: {e}")
        print("  The dataset may require authentication or have a different split name.")
        _save_datadecide_instructions()
        return None

    # Take a sample to inspect the schema
    print("  Inspecting schema...")
    try:
        sample = next(iter(ds))
        print(f"  Columns: {list(sample.keys())}")
    except Exception as e:
        print(f"  Cannot read sample: {e}")
        return None

    # Determine which columns hold the benchmark scores
    # DataDecide uses OLMES benchmarks — look for our target tasks
    score_cols = {}
    for bench in SHARED_BENCHMARKS:
        for col in sample.keys():
            if bench in col.lower() and ('acc' in col.lower()):
                score_cols[bench] = col
                break

    if not score_cols:
        print(f"  Cannot find benchmark columns. Available: {list(sample.keys())}")
        return None

    print(f"  Benchmark columns found: {score_cols}")

    # Identify the recipe and size columns
    recipe_col = next((k for k in sample.keys()
                        if 'recipe' in k.lower() or 'mix' in k.lower()
                        or 'data' in k.lower()), None)
    size_col   = next((k for k in sample.keys()
                        if 'size' in k.lower() or 'param' in k.lower()), None)
    step_col   = next((k for k in sample.keys()
                        if 'step' in k.lower()), None)
    model_col  = next((k for k in sample.keys()
                        if 'model' in k.lower()), None)

    print(f"  Key columns: recipe={recipe_col}, size={size_col}, "
          f"step={step_col}, model={model_col}")

    # Collect rows for dclm-baseline (or first available recipe)
    target_recipe = 'dclm-baseline'
    rows = []
    n_scanned = 0
    MAX_SCAN = 500000  # stop after scanning this many rows

    print(f"\n  Scanning for recipe='{target_recipe}'...")
    print(f"  (press Ctrl+C to stop early if taking too long)")

    try:
        for row in ds:
            n_scanned += 1
            if n_scanned % 10000 == 0:
                print(f"  Scanned {n_scanned} rows, found {len(rows)} matches...")

            if n_scanned > MAX_SCAN:
                print(f"  Reached scan limit ({MAX_SCAN})")
                break

            # Check recipe
            if recipe_col and row.get(recipe_col) != target_recipe:
                continue

            # Compute mean score
            scores = []
            for bench, col in score_cols.items():
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    scores.append(float(v))

            if len(scores) < 2:
                continue

            rows.append({
                'family': f'datadecide_{target_recipe}',
                'model': str(row.get(model_col or size_col, 'unknown')),
                'step': int(row.get(step_col, 0)) if step_col else 0,
                'score': float(np.mean(scores)),
                'size': str(row.get(size_col, '')),
            })

    except KeyboardInterrupt:
        print(f"\n  Interrupted after {n_scanned} rows")

    if not rows:
        print("  No rows found. Dataset schema may differ from expected.")
        _save_datadecide_instructions()
        return None

    df = pd.DataFrame(rows)

    # Select evenly spaced checkpoints per model
    final_rows = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model].sort_values('step')
        if len(mdf) >= n_checkpoints:
            indices = np.linspace(0, len(mdf)-1, n_checkpoints, dtype=int)
            final_rows.append(mdf.iloc[indices])
        else:
            final_rows.append(mdf)

    df_final = pd.concat(final_rows, ignore_index=True)
    out = os.path.join(OUT_DIR, 'evals_datadecide.csv')
    df_final.to_csv(out, index=False)
    print(f"\n  ✓ {len(df_final)} rows → {out}")
    _show_summary(df_final)
    return df_final


def _save_datadecide_instructions():
    txt = """
DataDecide load_dataset failed.

The dataset is at: https://huggingface.co/datasets/allenai/DataDecide-eval-results

Try:
  1. pip install datasets
  2. huggingface-cli login  (if gated)
  3. python extract_evals_v3.py --datadecide

Or download the parquet files manually from HuggingFace and read with:
  import pandas as pd
  df = pd.read_parquet('path/to/data-00000-of-NNNNN.parquet')
  print(df.columns.tolist())
  print(df.head(2))
Then share the column names and I'll update the extractor.
"""
    with open('datadecide_instructions.txt', 'w') as f:
        f.write(txt)
    print("  Instructions saved to datadecide_instructions.txt")


# ── Summary helper ────────────────────────────────────────────────────

def _show_summary(df):
    print(f"  Models: {df['model'].nunique()}")
    steps_per = df.groupby('model')['step'].count()
    print(f"  Steps per model: min={steps_per.min()}, max={steps_per.max()}")
    print(f"  Score range: {df['score'].min():.4f} — {df['score'].max():.4f}")


# ── Merge ─────────────────────────────────────────────────────────────

def merge_all():
    dfs = []
    for fname in ['evals_polypythia.csv', 'evals_datadecide.csv', 'evals_olmo.csv']:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  {len(df)} rows from {fname}")
    if not dfs:
        print("  No CSVs to merge.")
        return None
    merged = pd.concat(dfs, ignore_index=True)
    out = os.path.join(OUT_DIR, 'evals_all.csv')
    merged.to_csv(out, index=False)
    print(f"\n  ✓ {len(merged)} total rows → {out}")
    for fam in merged['family'].unique():
        n = len(merged[merged['family'] == fam]['model'].unique())
        print(f"    {fam}: {n} models")
    return merged


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--polypythia',  action='store_true')
    parser.add_argument('--datadecide',  action='store_true')
    parser.add_argument('--merge',       action='store_true')
    parser.add_argument('--all',         action='store_true')
    args = parser.parse_args()

    run_all = args.all or not any([args.polypythia, args.datadecide, args.merge])

    if args.merge:
        merge_all(); return

    if run_all or args.polypythia:
        extract_polypythia()

    if run_all or args.datadecide:
        extract_datadecide()

    print("\nMerging...")
    merge_all()
    print("\nDone. Run: python pt_alpha_sweep_multi.py")


if __name__ == '__main__':
    main()
