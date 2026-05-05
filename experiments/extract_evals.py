"""
Multi-Family Eval Data Extractor
=================================
Three extractors in one script. Each downloads pre-computed evaluation
scores for intermediate checkpoints and saves a CSV in the same format
that pt_alpha_sweep.py already consumes.

Output CSV format (same for all three):
  family, model, step, score

Where `score` is the mean accuracy across the shared benchmark subset
(piqa, arc_easy, arc_challenge, winogrande, lambada_openai where available).

Usage:
    # Run all three
    python extract_evals.py --all

    # Run specific sources
    python extract_evals.py --polypythia
    python extract_evals.py --datadecide
    python extract_evals.py --olmo

    # Install deps first:
    pip install datasets huggingface_hub pandas wandb --break-system-packages

Run from C:\\Users\\Carolina\\

Output files:
    ./evals_polypythia.csv
    ./evals_datadecide.csv
    ./evals_olmo.csv
    ./evals_all.csv   (merged, used by pt_alpha_sweep.py)
"""

import os
import json
import re
import argparse
import numpy as np
import pandas as pd

# ── Shared benchmark subset ───────────────────────────────────────────
# These 4 benchmarks exist in all three families.
# lambada_openai is in Pythia and OLMo but NOT DataDecide —
# we handle that per-family below.
SHARED_BENCHMARKS = ['piqa', 'arc_easy', 'arc_challenge', 'winogrande']

# ── Output path ───────────────────────────────────────────────────────
OUT_DIR = '.'


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 1: PolyPythias
# Same flat JSON structure as Pythia. Pre-computed evals on HuggingFace
# at EleutherAI/polypythias-evals.
# ═══════════════════════════════════════════════════════════════════════

def extract_polypythia(
    local_dir='./polypythias-evals',
    models=None,
    n_checkpoints=16,
):
    """
    Download PolyPythias evaluation results from HuggingFace and extract
    checkpoint score trajectories.

    Args:
        local_dir:      where to cache the downloaded files
        models:         list of model names to include (default: one seed
                        per size for clean per-scale comparison)
        n_checkpoints:  how many evenly-spaced checkpoints to sample
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Run: pip install huggingface_hub")
        return None

    # Default: one representative per size (seed 1)
    if models is None:
        models = [
            'pythia-14m-seed1',
            'pythia-70m-seed1',
            'pythia-160m-seed1',
            'pythia-410m-seed1',
        ]

    print(f"\n[PolyPythias] Downloading evals for {len(models)} models...")

    # Download only the folders we need (much smaller than full dataset)
    try:
        snapshot_download(
            repo_id='EleutherAI/polypythias-evals',
            repo_type='dataset',
            local_dir=local_dir,
            allow_patterns=[f"{m}/*" for m in models],
            ignore_patterns=['*.md'],
        )
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Trying with smaller subset...")

    rows = []
    for model in models:
        model_dir = os.path.join(local_dir, model)
        if not os.path.exists(model_dir):
            print(f"  Warning: {model} not found at {model_dir}")
            continue

        # Collect all available steps
        step_dirs = {}
        for entry in os.listdir(model_dir):
            m = re.match(r'step(\d+)', entry)
            if m:
                step_dirs[int(m.group(1))] = os.path.join(model_dir, entry)

        if not step_dirs:
            print(f"  No step dirs found for {model}")
            continue

        all_steps = sorted(step_dirs.keys())

        # Sample n_checkpoints evenly
        indices = np.linspace(0, len(all_steps) - 1, n_checkpoints, dtype=int)
        sampled = [all_steps[i] for i in indices]

        for step in sampled:
            step_dir = step_dirs[step]
            scores = []
            for bench in SHARED_BENCHMARKS:
                # PolyPythias uses same format as Pythia:
                # zero-shot/results_{step}.json or individual bench files
                # Try both patterns
                found = False
                for fname in os.listdir(step_dir):
                    if bench in fname.lower() and fname.endswith('.json'):
                        try:
                            with open(os.path.join(step_dir, fname)) as f:
                                data = json.load(f)
                            r = data.get('results', data)
                            if bench in r:
                                bd = r[bench]
                                s = bd.get('acc_norm', bd.get('acc', None))
                                if s is not None:
                                    scores.append(float(s))
                                    found = True
                                    break
                        except Exception:
                            pass
                if not found:
                    # Try aggregate results file
                    for fname in ['results.json', f'results_{step}.json']:
                        fpath = os.path.join(step_dir, fname)
                        if os.path.exists(fpath):
                            try:
                                with open(fpath) as f:
                                    data = json.load(f)
                                r = data.get('results', data)
                                if bench in r:
                                    bd = r[bench]
                                    s = bd.get('acc_norm', bd.get('acc', None))
                                    if s is not None:
                                        scores.append(float(s))
                                        break
                            except Exception:
                                pass

            if len(scores) >= 3:
                rows.append({
                    'family': 'polypythia',
                    'model': model,
                    'step': step,
                    'score': float(np.mean(scores)),
                    'n_tasks': len(scores),
                })

    if not rows:
        print("  No rows extracted. Check directory structure.")
        return None

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'evals_polypythia.csv')
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df)} rows to {out_path}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Steps per model: {df.groupby('model')['step'].count().to_dict()}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 2: DataDecide (AI2)
# HuggingFace dataset: allenai/DataDecide-eval-results
# Structure: models/{mix}/{size}/{seed}/{step}/{task}-metrics.json
# We pick one training recipe and sample across sizes to get scaling curves.
# ═══════════════════════════════════════════════════════════════════════

# DataDecide benchmark names (OLMES suite, slightly different from Pythia names)
DATADECIDE_BENCHMARKS = {
    'arc_challenge': ['acc_norm', 'acc'],
    'arc_easy':      ['acc_norm', 'acc'],
    'piqa':          ['acc_norm', 'acc'],
    'winogrande':    ['acc_norm', 'acc'],
    # hellaswag available but not in our shared subset — skip for comparability
}

def extract_datadecide(
    recipe='dclm-baseline',
    sizes=None,
    seed='seed-default',
    n_checkpoints=16,
    local_dir='./datadecide-evals',
):
    """
    Download DataDecide eval results for one training recipe across sizes.

    Args:
        recipe:         one of the 25 DataDecide training recipes
                        (dclm-baseline, dolma1_7, c4, fineweb-edu, etc.)
        sizes:          model sizes to include (default: range matching our
                        Pythia analysis)
        seed:           which seed to use (default: 'seed-default')
        n_checkpoints:  checkpoints to sample per model
        local_dir:      local cache directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Run: pip install huggingface_hub")
        return None

    # Default sizes: span several scales like Pythia's 7 configs
    if sizes is None:
        sizes = ['8M', '28M', '60M', '150M', '300M', '600M', '1B']

    print(f"\n[DataDecide] Recipe={recipe}, sizes={sizes}")

    patterns = [
        f"models/{recipe}/{sz}/{seed}/*" for sz in sizes
    ]

    try:
        snapshot_download(
            repo_id='allenai/DataDecide-eval-results',
            repo_type='dataset',
            local_dir=local_dir,
            allow_patterns=patterns,
            ignore_patterns=['*.md'],
        )
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Try: huggingface-cli login  then re-run")
        return None

    rows = []
    base = os.path.join(local_dir, 'models', recipe)
    if not os.path.exists(base):
        print(f"  Base dir not found: {base}")
        return None

    for sz in sizes:
        sz_dir = os.path.join(base, sz, seed)
        if not os.path.exists(sz_dir):
            print(f"  Missing: {sz_dir}")
            continue

        step_dirs = {}
        for entry in os.listdir(sz_dir):
            m = re.match(r'step[_-]?(\d+)', entry)
            if m:
                step_dirs[int(m.group(1))] = os.path.join(sz_dir, entry)

        if not step_dirs:
            print(f"  No step dirs for {recipe}/{sz}")
            continue

        all_steps = sorted(step_dirs.keys())
        indices = np.linspace(0, len(all_steps) - 1, n_checkpoints, dtype=int)
        sampled = [all_steps[i] for i in indices]

        for step in sampled:
            step_dir = step_dirs[step]
            scores = []
            for bench, metric_keys in DATADECIDE_BENCHMARKS.items():
                # DataDecide saves individual {bench}-metrics.json per task
                for fname in [f'{bench}-metrics.json',
                               f'{bench}_metrics.json',
                               f'{bench}.json']:
                    fpath = os.path.join(step_dir, fname)
                    if os.path.exists(fpath):
                        try:
                            with open(fpath) as f:
                                data = json.load(f)
                            # Try various key patterns
                            score = None
                            for mkey in metric_keys:
                                score = (data.get(mkey) or
                                         data.get('metrics', {}).get(mkey) or
                                         data.get('results', {}).get(bench, {}).get(mkey))
                                if score is not None:
                                    break
                            if score is not None:
                                scores.append(float(score))
                                break
                        except Exception:
                            pass

            if len(scores) >= 3:
                rows.append({
                    'family': f'datadecide_{recipe}',
                    'model': f'{recipe}-{sz}',
                    'step': step,
                    'score': float(np.mean(scores)),
                    'n_tasks': len(scores),
                })

    if not rows:
        print("  No rows extracted. Check directory and schema.")
        return None

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'evals_datadecide.csv')
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df)} rows to {out_path}")
    print(f"  Models: {df['model'].unique().tolist()}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 3: OLMo via WandB API
# Public runs at wandb.ai/ai2-llm. Requires free WandB account.
# Extract per-step benchmark metrics from training logs.
# ═══════════════════════════════════════════════════════════════════════

# WandB metric key patterns for OLMo evaluations.
# OLMo logs eval metrics during training — exact key names vary by run.
OLMO_METRIC_PATTERNS = [
    # Pattern 1: eval/{benchmark}/acc
    ('piqa',         ['eval/piqa/acc_norm', 'eval/piqa/acc', 'piqa_acc']),
    ('arc_easy',     ['eval/arc_easy/acc_norm', 'eval/arc_easy/acc', 'arc_easy_acc']),
    ('arc_challenge',['eval/arc_challenge/acc_norm', 'eval/arc_challenge/acc']),
    ('winogrande',   ['eval/winogrande/acc_norm', 'eval/winogrande/acc']),
]

# Known public OLMo WandB run IDs (as of May 2026)
# Format: (model_name, entity/project/run_id)
OLMO_RUNS = [
    ('olmo-1b',  'ai2-llm/OLMo/olmo-1b'),   # OLMo 1B training run
    ('olmo-7b',  'ai2-llm/OLMo/olmo-7b'),   # OLMo 7B training run
]

def extract_olmo(
    n_checkpoints=16,
    out_path=None,
):
    """
    Extract OLMo benchmark metrics from public WandB training runs.
    Requires: pip install wandb + free account at wandb.ai

    If WandB is not available or runs are not found, saves a download
    instruction file instead.
    """
    try:
        import wandb
    except ImportError:
        print("[OLMo] wandb not installed. Run: pip install wandb")
        _save_olmo_instructions()
        return None

    print("\n[OLMo] Extracting from WandB public runs...")
    print("  If this fails, see: olmo_download_instructions.txt")

    api = wandb.Api(timeout=60)
    rows = []

    for model_name, run_path in OLMO_RUNS:
        print(f"  Trying run: {run_path}")
        try:
            run = api.run(run_path)

            # Get available keys to find the right metric names
            history_keys = list(run.history(samples=1).columns)
            print(f"    Available keys (sample): {history_keys[:10]}")

            # Find which metric keys actually exist in this run
            active_metrics = {}
            for bench, candidate_keys in OLMO_METRIC_PATTERNS:
                for k in candidate_keys:
                    if k in history_keys:
                        active_metrics[bench] = k
                        break

            if not active_metrics:
                print(f"    No matching metric keys found for {model_name}")
                print(f"    All keys: {history_keys}")
                continue

            print(f"    Using metrics: {active_metrics}")

            # Pull full history for active metrics
            keys = list(active_metrics.values()) + ['_step']
            history = run.history(keys=keys, samples=10000)
            history = history.dropna(subset=list(active_metrics.values()))

            if len(history) == 0:
                print(f"    No history rows found")
                continue

            # Sample n_checkpoints evenly
            all_steps = sorted(history['_step'].unique())
            indices = np.linspace(0, len(all_steps) - 1,
                                   n_checkpoints, dtype=int)
            sampled_steps = [all_steps[i] for i in indices]

            for step in sampled_steps:
                row_data = history[history['_step'] == step]
                if row_data.empty:
                    continue
                scores = []
                for bench, key in active_metrics.items():
                    val = row_data[key].values[0]
                    if not np.isnan(val):
                        scores.append(float(val))
                if len(scores) >= 3:
                    rows.append({
                        'family': 'olmo',
                        'model': model_name,
                        'step': int(step),
                        'score': float(np.mean(scores)),
                        'n_tasks': len(scores),
                    })

            print(f"    ✓ {model_name}: {len([r for r in rows if r['model']==model_name])} steps")

        except wandb.errors.CommError as e:
            print(f"    CommError: {e}")
            print("    Run may be private or run_id may have changed.")
        except Exception as e:
            print(f"    Error: {e}")

    if not rows:
        print("  No rows extracted from WandB.")
        _save_olmo_instructions()
        return None

    df = pd.DataFrame(rows)
    path = out_path or os.path.join(OUT_DIR, 'evals_olmo.csv')
    df.to_csv(path, index=False)
    print(f"  ✓ Saved {len(df)} rows to {path}")
    return df


def _save_olmo_instructions():
    """Write a fallback instructions file if WandB extraction fails."""
    instructions = """
OLMo Eval Extraction — Manual Fallback
=======================================

If the WandB API extraction failed, try these steps:

1. Create a free account at https://wandb.ai
2. Install: pip install wandb
3. Login: wandb login
4. Re-run: python extract_evals.py --olmo

If the run IDs are stale (they change), find them at:
  https://wandb.ai/ai2-llm

Look for:
  - "OLMo-7B" training run (search "OLMo 7B" in projects)
  - "OLMo-1B" training run

The metric keys to look for in run history:
  - eval/piqa/acc or eval/piqa/acc_norm
  - eval/arc_easy/acc or eval/arc_easy/acc_norm
  - eval/arc_challenge/acc
  - eval/winogrande/acc

Manual CSV format to produce (if needed):
  family,model,step,score,n_tasks
  olmo,olmo-7b,1000,0.45,4
  olmo,olmo-7b,5000,0.51,4
  ...

Save as: evals_olmo.csv
Then run: python pt_alpha_sweep_multi.py
"""
    with open('olmo_download_instructions.txt', 'w') as f:
        f.write(instructions)
    print("  Saved fallback instructions to olmo_download_instructions.txt")


# ═══════════════════════════════════════════════════════════════════════
# Merge and validate
# ═══════════════════════════════════════════════════════════════════════

def merge_all():
    """Merge all extracted CSVs into one for pt_alpha_sweep_multi.py."""
    dfs = []
    for fname in ['evals_polypythia.csv', 'evals_datadecide.csv',
                   'evals_olmo.csv']:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {fname}")
        else:
            print(f"  Not found: {fname}")

    if not dfs:
        print("  No CSVs found to merge.")
        return None

    merged = pd.concat(dfs, ignore_index=True)
    out = os.path.join(OUT_DIR, 'evals_all.csv')
    merged.to_csv(out, index=False)
    print(f"\n  ✓ Merged {len(merged)} total rows to {out}")
    print(f"  Families: {merged['family'].unique().tolist()}")
    print(f"  Models per family:")
    for fam in merged['family'].unique():
        models = merged[merged['family']==fam]['model'].unique()
        print(f"    {fam}: {list(models)}")
    return merged


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Extract multi-family eval data for Paper 3 generalization')
    parser.add_argument('--all', action='store_true',
                        help='Run all three extractors')
    parser.add_argument('--polypythia', action='store_true',
                        help='Extract PolyPythias evals')
    parser.add_argument('--datadecide', action='store_true',
                        help='Extract DataDecide evals')
    parser.add_argument('--olmo', action='store_true',
                        help='Extract OLMo evals via WandB')
    parser.add_argument('--merge', action='store_true',
                        help='Merge existing CSVs only')
    parser.add_argument('--recipe', default='dclm-baseline',
                        help='DataDecide training recipe (default: dclm-baseline)')
    args = parser.parse_args()

    if args.merge:
        merge_all()
        return

    run_all = args.all or not any([args.polypythia, args.datadecide, args.olmo])

    if run_all or args.polypythia:
        print("=" * 60)
        print("SOURCE 1: PolyPythias (EleutherAI, same format as Pythia)")
        print("=" * 60)
        extract_polypythia()

    if run_all or args.datadecide:
        print("\n" + "=" * 60)
        print(f"SOURCE 2: DataDecide (AI2, recipe={args.recipe})")
        print("=" * 60)
        extract_datadecide(recipe=args.recipe)

    if run_all or args.olmo:
        print("\n" + "=" * 60)
        print("SOURCE 3: OLMo (AI2, via WandB API)")
        print("=" * 60)
        extract_olmo()

    print("\n" + "=" * 60)
    print("Merging all available sources...")
    print("=" * 60)
    merge_all()

    print("""
Next step:
    python pt_alpha_sweep_multi.py

This will run the full α sweep on all downloaded families and tell you
whether α=0.6 generalizes beyond Pythia.
""")


if __name__ == '__main__':
    main()
