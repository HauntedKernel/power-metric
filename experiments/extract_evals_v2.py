"""
Multi-Family Eval Extractor — Rate-Limit-Friendly Version
==========================================================
Fixes HTTP 429 errors by fetching only the specific checkpoint
files we need (16 per model) instead of bulk snapshot_download.

Usage:
    python extract_evals_v2.py --polypythia
    python extract_evals_v2.py --datadecide
    python extract_evals_v2.py --olmo
    python extract_evals_v2.py --all
    python extract_evals_v2.py --merge

Install deps:
    pip install datasets huggingface_hub pandas wandb --break-system-packages

Run from C:\\Users\\Carolina\\
"""

import os
import json
import re
import time
import argparse
import numpy as np
import pandas as pd

OUT_DIR = '.'
SHARED_BENCHMARKS = ['piqa', 'arc_easy', 'arc_challenge', 'winogrande']
DELAY_BETWEEN_FILES = 1.5   # seconds between HF requests
DELAY_ON_429 = 30.0         # seconds to wait on rate limit error


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def safe_hf_download(repo_id, filename, repo_type='dataset',
                      local_dir='.', retries=5):
    """Download a single file from HuggingFace with retry on 429."""
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
            time.sleep(DELAY_BETWEEN_FILES)
            return path
        except Exception as e:
            msg = str(e)
            if '429' in msg or 'rate limit' in msg.lower():
                wait = DELAY_ON_429 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait:.0f}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                print(f"    Download error: {e}")
                return None
    print(f"    Failed after {retries} attempts: {filename}")
    return None


def score_from_json(data, benchmarks=SHARED_BENCHMARKS):
    """Extract mean accuracy from an eval results JSON."""
    scores = []
    r = data.get('results', data)
    for bench in benchmarks:
        if bench in r:
            bd = r[bench]
            s = bd.get('acc_norm', bd.get('acc', None))
            if s is not None:
                scores.append(float(s))
    return float(np.mean(scores)) if len(scores) >= 2 else None


# ══════════════════════════════════════════════════════════════════════
# SOURCE 1: PolyPythias
# repo: EleutherAI/polypythias-evals
# structure: {model}/step{N}/EleutherAI__{model}%2F... .json
# We need to list what files are available first, then pick 16 steps.
# ══════════════════════════════════════════════════════════════════════

def list_polypythia_steps(model, cache_dir='./polypythias-evals'):
    """List available step numbers for a PolyPythias model."""
    try:
        from huggingface_hub import list_repo_tree
        files = list_repo_tree(
            repo_id='EleutherAI/polypythias-evals',
            repo_type='dataset',
            path_in_repo=model,
            recursive=False,
        )
        steps = []
        for f in files:
            m = re.match(r'step(\d+)', f.path.split('/')[-1])
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)
    except Exception as e:
        print(f"  Could not list steps for {model}: {e}")
        return []


def extract_polypythia(n_checkpoints=16, cache_dir='./polypythias-evals'):
    """
    Download only the specific step files we need for PolyPythias.
    ~64 files total (4 models × 16 steps) vs 2400+ with snapshot_download.
    """
    try:
        from huggingface_hub import list_repo_tree
    except ImportError:
        print("Run: pip install huggingface_hub")
        return None

    models = [
        'pythia-14m-seed1',
        'pythia-70m-seed1',
        'pythia-160m-seed1',
        'pythia-410m-seed1',
    ]

    print(f"\n[PolyPythias] Fetching {n_checkpoints} checkpoints for {len(models)} models")
    print(f"  Targeted download: ~{n_checkpoints * len(models)} files (not bulk)")

    rows = []

    for model in models:
        print(f"\n  Model: {model}")

        # Step 1: list available steps (one API call)
        steps = list_polypythia_steps(model, cache_dir)
        if not steps:
            print(f"    No steps found")
            continue
        print(f"    Found {len(steps)} steps: {steps[0]}...{steps[-1]}")

        # Step 2: pick evenly spaced subset
        indices = np.linspace(0, len(steps) - 1, n_checkpoints, dtype=int)
        sampled = [steps[i] for i in indices]

        # Step 3: download just those step dirs
        for step in sampled:
            # Need to list files within the step dir to get filename
            try:
                step_files = list(list_repo_tree(
                    repo_id='EleutherAI/polypythias-evals',
                    repo_type='dataset',
                    path_in_repo=f'{model}/step{step}',
                    recursive=False,
                ))
                time.sleep(0.5)
            except Exception as e:
                print(f"    step{step}: listing failed: {e}")
                continue

            # Find the results JSON file
            target = None
            for f in step_files:
                fname = f.path.split('/')[-1]
                if fname.endswith('.json') and 'results' in fname:
                    target = f.path
                    break

            if not target:
                print(f"    step{step}: no results JSON found")
                continue

            # Download it
            local_path = safe_hf_download(
                repo_id='EleutherAI/polypythias-evals',
                filename=target,
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
                        print(f"    step{step}: {score:.4f} ✓")
                except Exception as e:
                    print(f"    step{step}: parse error: {e}")

    if not rows:
        print("\n  No rows extracted.")
        return None

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, 'evals_polypythia.csv')
    df.to_csv(out, index=False)
    print(f"\n  ✓ Saved {len(df)} rows → {out}")
    return df


# ══════════════════════════════════════════════════════════════════════
# SOURCE 2: DataDecide
# repo: allenai/DataDecide-eval-results
# structure: models/{recipe}/{size}/{seed}/{step}/{bench}-metrics.json
# ══════════════════════════════════════════════════════════════════════

DATADECIDE_BENCHMARKS = {
    'arc_challenge': ['acc_norm', 'acc'],
    'arc_easy':      ['acc_norm', 'acc'],
    'piqa':          ['acc_norm', 'acc'],
    'winogrande':    ['acc_norm', 'acc'],
}

def extract_datadecide(
    recipe='dclm-baseline',
    sizes=None,
    seed='seed-default',
    n_checkpoints=16,
    cache_dir='./datadecide-evals',
):
    """
    Download targeted DataDecide eval files — one benchmark file per
    checkpoint × size instead of bulk.
    ~64 target files for 4 sizes × 16 checkpoints.
    """
    try:
        from huggingface_hub import list_repo_tree
    except ImportError:
        print("Run: pip install huggingface_hub")
        return None

    if sizes is None:
        sizes = ['28M', '60M', '150M', '300M']

    print(f"\n[DataDecide] recipe={recipe}, sizes={sizes}")
    print(f"  Targeted download: ~{n_checkpoints * len(sizes) * len(DATADECIDE_BENCHMARKS)} files")

    rows = []

    for sz in sizes:
        print(f"\n  Size: {sz}")
        base_path = f'models/{recipe}/{sz}/{seed}'

        # List step directories
        try:
            step_entries = list(list_repo_tree(
                repo_id='allenai/DataDecide-eval-results',
                repo_type='dataset',
                path_in_repo=base_path,
                recursive=False,
            ))
            time.sleep(0.5)
        except Exception as e:
            print(f"    Cannot list {base_path}: {e}")
            continue

        steps = {}
        for entry in step_entries:
            name = entry.path.split('/')[-1]
            m = re.match(r'step[_-]?(\d+)', name)
            if m:
                steps[int(m.group(1))] = f'{base_path}/{name}'

        if not steps:
            print(f"    No step dirs found")
            continue

        all_steps = sorted(steps.keys())
        print(f"    {len(all_steps)} steps: {all_steps[0]}...{all_steps[-1]}")

        indices = np.linspace(0, len(all_steps) - 1, n_checkpoints, dtype=int)
        sampled = [all_steps[i] for i in indices]

        for step in sampled:
            step_path = steps[step]
            step_scores = []

            for bench, metric_keys in DATADECIDE_BENCHMARKS.items():
                # Try different filename patterns
                for fname_pattern in [f'{bench}-metrics.json',
                                       f'{bench}_metrics.json',
                                       f'{bench}.json']:
                    filepath = f'{step_path}/{fname_pattern}'
                    local_path = safe_hf_download(
                        repo_id='allenai/DataDecide-eval-results',
                        filename=filepath,
                        repo_type='dataset',
                        local_dir=cache_dir,
                    )
                    if local_path and os.path.exists(local_path):
                        try:
                            with open(local_path) as f:
                                data = json.load(f)
                            score = None
                            for mkey in metric_keys:
                                score = (data.get(mkey) or
                                         data.get('metrics', {}).get(mkey) or
                                         data.get('results', {}).get(bench, {}).get(mkey))
                                if score is not None:
                                    break
                            if score is not None:
                                step_scores.append(float(score))
                                break
                        except Exception:
                            pass

            if len(step_scores) >= 3:
                rows.append({
                    'family': f'datadecide_{recipe}',
                    'model': f'{recipe}-{sz}',
                    'step': step,
                    'score': float(np.mean(step_scores)),
                })
                print(f"    step{step}: {np.mean(step_scores):.4f} ({len(step_scores)} tasks) ✓")
            else:
                print(f"    step{step}: only {len(step_scores)} tasks, skipping")

    if not rows:
        print("\n  No rows extracted.")
        return None

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, 'evals_datadecide.csv')
    df.to_csv(out, index=False)
    print(f"\n  ✓ Saved {len(df)} rows → {out}")
    return df


# ══════════════════════════════════════════════════════════════════════
# SOURCE 3: OLMo via WandB
# (unchanged from previous version — WandB doesn't have rate limit issue)
# ══════════════════════════════════════════════════════════════════════

OLMO_METRIC_PATTERNS = [
    ('piqa',         ['eval/piqa/acc_norm', 'eval/piqa/acc']),
    ('arc_easy',     ['eval/arc_easy/acc_norm', 'eval/arc_easy/acc']),
    ('arc_challenge',['eval/arc_challenge/acc_norm', 'eval/arc_challenge/acc']),
    ('winogrande',   ['eval/winogrande/acc_norm', 'eval/winogrande/acc']),
]

def extract_olmo(n_checkpoints=16):
    try:
        import wandb
    except ImportError:
        print("[OLMo] Run: pip install wandb && wandb login")
        _save_olmo_instructions()
        return None

    print("\n[OLMo] Extracting from WandB public runs...")
    api = wandb.Api(timeout=60)
    rows = []

    # Try to find OLMo runs in the public ai2-llm project
    try:
        runs = api.runs("ai2-llm", filters={"display_name": {"$regex": "OLMo"}})
        print(f"  Found {len(list(runs))} OLMo runs")
    except Exception as e:
        print(f"  Could not search runs: {e}")
        print("  See olmo_download_instructions.txt for manual path")
        _save_olmo_instructions()
        return None

    for run in api.runs("ai2-llm", filters={"display_name": {"$regex": "OLMo"}}):
        print(f"  Run: {run.name} ({run.id})")
        try:
            history_keys = list(run.history(samples=1).columns)
            active = {}
            for bench, candidates in OLMO_METRIC_PATTERNS:
                for k in candidates:
                    if k in history_keys:
                        active[bench] = k
                        break

            if len(active) < 2:
                continue

            keys = list(active.values()) + ['_step']
            history = run.history(keys=keys, samples=10000)
            history = history.dropna(subset=list(active.values()))
            if len(history) == 0:
                continue

            all_steps = sorted(history['_step'].unique())
            indices = np.linspace(0, len(all_steps)-1, n_checkpoints, dtype=int)
            sampled = [all_steps[i] for i in indices]

            for step in sampled:
                rdata = history[history['_step'] == step]
                if rdata.empty: continue
                scores = []
                for bench, key in active.items():
                    val = rdata[key].values[0]
                    if not np.isnan(val):
                        scores.append(float(val))
                if len(scores) >= 3:
                    rows.append({
                        'family': 'olmo',
                        'model': run.name,
                        'step': int(step),
                        'score': float(np.mean(scores)),
                    })
            print(f"    ✓ {run.name}: extracted rows")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    if not rows:
        print("  No rows extracted.")
        _save_olmo_instructions()
        return None

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, 'evals_olmo.csv')
    df.to_csv(out, index=False)
    print(f"  ✓ Saved {len(df)} rows → {out}")
    return df


def _save_olmo_instructions():
    txt = """
OLMo WandB extraction failed. Manual steps:
1. Create account: https://wandb.ai
2. pip install wandb && wandb login
3. Go to https://wandb.ai/ai2-llm, find an OLMo training run
4. Note the run path (entity/project/run_id)
5. Edit OLMO_RUNS in this script and re-run
"""
    with open('olmo_download_instructions.txt', 'w') as f:
        f.write(txt)


# ══════════════════════════════════════════════════════════════════════
# Merge
# ══════════════════════════════════════════════════════════════════════

def merge_all():
    dfs = []
    for fname in ['evals_polypythia.csv', 'evals_datadecide.csv', 'evals_olmo.csv']:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {fname}")
    if not dfs:
        print("  No CSVs to merge.")
        return None
    merged = pd.concat(dfs, ignore_index=True)
    out = os.path.join(OUT_DIR, 'evals_all.csv')
    merged.to_csv(out, index=False)
    print(f"  ✓ Merged {len(merged)} rows → {out}")
    return merged


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all',         action='store_true')
    parser.add_argument('--polypythia',  action='store_true')
    parser.add_argument('--datadecide',  action='store_true')
    parser.add_argument('--olmo',        action='store_true')
    parser.add_argument('--merge',       action='store_true')
    parser.add_argument('--recipe',      default='dclm-baseline')
    args = parser.parse_args()

    if args.merge:
        merge_all(); return

    run_all = args.all or not any([args.polypythia, args.datadecide, args.olmo])

    if run_all or args.polypythia:
        extract_polypythia()

    if run_all or args.datadecide:
        extract_datadecide(recipe=args.recipe)

    if run_all or args.olmo:
        extract_olmo()

    print("\nMerging...")
    merge_all()
    print("\nDone. Run: python pt_alpha_sweep_multi.py")


if __name__ == '__main__':
    main()
