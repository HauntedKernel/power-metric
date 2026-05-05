"""
Schema Inspector — prints actual content of downloaded files
============================================================
Run this to show the real JSON structure of PolyPythias files
and the real schema of DataDecide.

Run from C:\\Users\\Carolina\\
"""

import os, json
import glob


def inspect_polypythia():
    print("=" * 70)
    print("PolyPythias JSON structure")
    print("=" * 70)

    # Find any downloaded results JSON
    cache_dir = './polypythias-evals'
    found = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.startswith('results_') and f.endswith('.json'):
                found.append(os.path.join(root, f))

    if not found:
        print("No cached files found. Looking in current directory...")
        found = glob.glob('./**/*results_*.json', recursive=True)

    if not found:
        print("No PolyPythias JSON files found locally.")
        print("Run: python extract_evals_v3.py --polypythia  first")
        return

    # Print first file structure
    for fpath in found[:2]:
        print(f"\nFile: {fpath}")
        print(f"Size: {os.path.getsize(fpath)} bytes")
        try:
            with open(fpath) as f:
                data = json.load(f)

            print(f"Top-level keys: {list(data.keys())}")

            if 'results' in data:
                results = data['results']
                print(f"\nresults keys (first 10): {list(results.keys())[:10]}")
                for task_key in list(results.keys())[:3]:
                    print(f"\n  results['{task_key}']:")
                    print(f"    {results[task_key]}")

            elif 'metrics' in data:
                print(f"\nmetrics: {data['metrics']}")

            else:
                # Print full structure for small files
                if os.path.getsize(fpath) < 20000:
                    import pprint
                    pprint.pprint(data, depth=3)
                else:
                    print(f"\nFull structure (truncated):")
                    for k, v in data.items():
                        if isinstance(v, dict):
                            print(f"  {k}: dict with keys {list(v.keys())[:5]}")
                        elif isinstance(v, list):
                            print(f"  {k}: list of {len(v)} items")
                        else:
                            print(f"  {k}: {str(v)[:80]}")
        except Exception as e:
            print(f"  Error reading: {e}")


def inspect_datadecide():
    print("\n" + "=" * 70)
    print("DataDecide schema — first 3 rows")
    print("=" * 70)
    try:
        from datasets import load_dataset
        ds = load_dataset(
            'allenai/DataDecide-eval-results',
            split='train',
            streaming=True,
        )
        print("\nFirst 3 rows:")
        for i, row in enumerate(ds):
            if i >= 3: break
            print(f"\nRow {i}:")
            for k, v in row.items():
                print(f"  {k}: {repr(v)[:100]}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == '__main__':
    inspect_polypythia()
    inspect_datadecide()
