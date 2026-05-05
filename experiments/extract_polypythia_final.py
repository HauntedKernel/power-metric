"""
PolyPythias Extractor — Final Version
======================================
Fixes:
  1. Metric key is 'acc,none' not 'acc' (newer lm-eval format)
  2. Scans what benchmarks are actually present and finds our targets
  3. Targets the 13KB summary files (September 2024 re-run), not 82KB
     full-task files (those have blimp etc., not our 4 benchmarks)

Run from C:\\Users\\Carolina\\
"""

import os, json, re, time
import numpy as np
import pandas as pd
from huggingface_hub import list_repo_tree, hf_hub_download

CACHE_DIR   = './polypythias-evals'
OUT_PATH    = './evals_polypythia.csv'
DELAY       = 1.2
DELAY_429   = 30.0
N_CKPTS     = 16

MODELS = [
    'pythia-14m-seed1',
    'pythia-70m-seed1',
    'pythia-160m-seed1',
    'pythia-410m-seed1',
]

# The 4 benchmarks we want — PolyPythias uses lm-eval newer format
# so keys are like 'piqa' but metric subkey is 'acc,none' or 'acc_norm,none'
TARGET_TASKS = ['piqa', 'arc_easy', 'arc_challenge', 'winogrande']


def safe_get(repo_id, filepath, retries=5):
    for attempt in range(retries):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type='dataset',
                local_dir=CACHE_DIR,
            )
            time.sleep(DELAY)
            return path
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                wait = DELAY_429 * (attempt + 1)
                print(f"    Rate limit. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif '404' in str(e):
                return None
            else:
                return None
    return None


def score_from_json(data):
    """
    Extract mean accuracy from PolyPythias results JSON.
    New lm-eval format: results[task]['acc,none'] or ['acc_norm,none']
    """
    results = data.get('results', {})
    scores = []
    for task in TARGET_TASKS:
        if task not in results:
            continue
        bd = results[task]
        # Try metric keys in order of preference
        for mkey in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
            if mkey in bd:
                scores.append(float(bd[mkey]))
                break
    return float(np.mean(scores)) if len(scores) >= 2 else None


def get_best_file(step_path, model):
    """
    List files in a step's subfolder and return path to the
    13KB summary file (September 2024 run with our 4 benchmarks).
    Avoids the 82KB blimp-heavy files.
    """
    subfolder = f"{step_path}/EleutherAI__{model}"
    try:
        files = list(list_repo_tree(
            repo_id='EleutherAI/polypythias-evals',
            repo_type='dataset',
            path_in_repo=subfolder,
            recursive=False,
        ))
        time.sleep(0.3)
    except Exception:
        return None

    json_files = sorted(
        [(getattr(f, 'size', 999999), f.path) for f in files if f.path.endswith('.json')]
    )

    if not json_files:
        return None

    # The 13KB files are from the September 2024 re-run with our 4 benchmarks
    # The 82KB files are the full blimp suite — skip those
    for size, path in json_files:
        if size < 20000:      # summary file
            return path, size

    # If only large files exist, return the smallest anyway
    return json_files[0][1], json_files[0][0]


def main():
    print("PolyPythias Extractor — correct key format (acc,none)")
    print()

    # First: inspect one cached file to confirm benchmark presence
    cached = []
    for root, _, files in os.walk(CACHE_DIR):
        for f in files:
            if f.startswith('results_') and f.endswith('.json'):
                cached.append(os.path.join(root, f))

    if cached:
        print(f"Found {len(cached)} cached files. Checking one for benchmark keys...")
        with open(cached[0]) as f:
            sample = json.load(f)
        results_keys = list(sample.get('results', {}).keys())
        has_targets = [t for t in TARGET_TASKS if t in results_keys]
        missing = [t for t in TARGET_TASKS if t not in results_keys]
        print(f"  Targets found: {has_targets}")
        print(f"  Targets missing: {missing}")
        if missing:
            print()
            print("  NOTE: These files are the 82KB blimp-suite runs.")
            print("  We need the 13KB files from September 2024 that include")
            print("  piqa/arc_easy/arc_challenge/winogrande.")
            print("  Downloading those specifically now...")
        print()

    rows = []

    for model in MODELS:
        print(f"Model: {model}")

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
            print(f"  Cannot list steps: {e}")
            continue

        steps = {}
        for entry in step_entries:
            name = entry.path.split('/')[-1]
            m = re.match(r'step(\d+)$', name)
            if m:
                steps[int(m.group(1))] = entry.path

        valid = sorted(s for s in steps if s >= 1000)
        if not valid:
            print(f"  No steps >= 1000")
            continue

        indices = np.linspace(0, len(valid)-1, N_CKPTS, dtype=int)
        sampled = [valid[i] for i in indices]
        print(f"  Steps: {sampled[0]}...{sampled[-1]} ({len(sampled)} samples)")

        for step in sampled:
            result = get_best_file(steps[step], model)
            if result is None:
                print(f"  step{step}: no files found")
                continue

            target_path, size = result
            print(f"  step{step}: downloading {size}B file...")

            local = safe_get('EleutherAI/polypythias-evals', target_path)
            if not local or not os.path.exists(local):
                print(f"  step{step}: download failed")
                continue

            with open(local) as f:
                data = json.load(f)

            score = score_from_json(data)
            if score is not None:
                rows.append({
                    'family': 'polypythia',
                    'model': model,
                    'step': step,
                    'score': score,
                })
                print(f"  step{step}: {score:.4f} ✓")
            else:
                # Print what tasks ARE in this file to help debug
                found_tasks = [k for k in data.get('results', {}).keys()
                                if not k.startswith(' ')][:8]
                print(f"  step{step}: targets not in file. Found: {found_tasks[:5]}")

    if not rows:
        print("\nNo rows extracted.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✓ {len(df)} rows → {OUT_PATH}")
    print(df.groupby('model')['step'].count().to_string())


if __name__ == '__main__':
    main()
