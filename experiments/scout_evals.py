"""
Directory Scout — Map actual HuggingFace dataset structures
============================================================
Run this BEFORE extract_evals_v3.py to confirm directory layouts.
Prints the actual folder/file names so we stop guessing.

Usage:
    python scout_evals.py

Run from C:\\Users\\Carolina\\
"""

import time
from huggingface_hub import list_repo_tree

DELAY = 0.5  # seconds between API calls


def tree(repo_id, path, repo_type='dataset', max_depth=3, _depth=0):
    """Print directory tree for a HuggingFace repo path."""
    try:
        entries = list(list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path,
            recursive=False,
        ))
        time.sleep(DELAY)
    except Exception as e:
        print(f"  {'  '*_depth}[ERROR: {e}]")
        return

    for entry in entries[:8]:  # cap at 8 to avoid spam
        name = entry.path.split('/')[-1]
        is_dir = getattr(entry, 'type', None) == 'directory' or not '.' in name
        size = getattr(entry, 'size', '')
        size_str = f" ({size} bytes)" if size else ""
        print(f"  {'  '*_depth}{'📁' if is_dir else '📄'} {name}{size_str}")
        if is_dir and _depth < max_depth - 1:
            tree(repo_id, entry.path, repo_type, max_depth, _depth + 1)

    if len(entries) > 8:
        print(f"  {'  '*_depth}... ({len(entries) - 8} more)")


def main():
    # ── PolyPythias ──────────────────────────────────────────────────
    print("=" * 70)
    print("PolyPythias — EleutherAI/polypythias-evals")
    print("=" * 70)

    print("\n  Top level:")
    tree('EleutherAI/polypythias-evals', '', max_depth=1)

    print("\n  One model, one step (drill down to see JSON path):")
    tree('EleutherAI/polypythias-evals',
         'pythia-160m-seed1',
         max_depth=3)

    # ── DataDecide ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DataDecide — allenai/DataDecide-eval-results")
    print("=" * 70)

    print("\n  Top level:")
    tree('allenai/DataDecide-eval-results', '', max_depth=1)

    print("\n  What's inside 'models'?")
    tree('allenai/DataDecide-eval-results', 'models', max_depth=1)

    print("\n  What recipes exist?")
    try:
        recipes = list(list_repo_tree(
            repo_id='allenai/DataDecide-eval-results',
            repo_type='dataset',
            path_in_repo='models',
            recursive=False,
        ))
        for r in recipes[:10]:
            name = r.path.split('/')[-1]
            print(f"    - {name}")
        if len(recipes) > 10:
            print(f"    ... ({len(recipes)-10} more)")
        time.sleep(DELAY)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n  Drill into dclm-baseline (first 2 levels):")
    tree('allenai/DataDecide-eval-results', 'models/dclm-baseline', max_depth=2)

    print("\n  One size, one seed, one step — full path to JSON:")
    # Try to find the first available size
    try:
        sizes = list(list_repo_tree(
            repo_id='allenai/DataDecide-eval-results',
            repo_type='dataset',
            path_in_repo='models/dclm-baseline',
            recursive=False,
        ))
        time.sleep(DELAY)
        if sizes:
            first_size = sizes[0].path
            print(f"\n  Drilling into: {first_size}")
            seeds = list(list_repo_tree(
                repo_id='allenai/DataDecide-eval-results',
                repo_type='dataset',
                path_in_repo=first_size,
                recursive=False,
            ))
            time.sleep(DELAY)
            if seeds:
                first_seed = seeds[0].path
                steps = list(list_repo_tree(
                    repo_id='allenai/DataDecide-eval-results',
                    repo_type='dataset',
                    path_in_repo=first_seed,
                    recursive=False,
                ))
                time.sleep(DELAY)
                if steps:
                    first_step = steps[0].path
                    print(f"  Step path: {first_step}")
                    files = list(list_repo_tree(
                        repo_id='allenai/DataDecide-eval-results',
                        repo_type='dataset',
                        path_in_repo=first_step,
                        recursive=False,
                    ))
                    for f in files[:10]:
                        print(f"    📄 {f.path.split('/')[-1]}")
    except Exception as e:
        print(f"  Error drilling in: {e}")

    print("\n" + "=" * 70)
    print("Done. Copy the actual paths above into extract_evals_v3.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
