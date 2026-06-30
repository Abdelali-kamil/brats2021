#!/usr/bin/env python3
"""
generate_test_ids.py

Creates a test_ids file from your dataset.

Usage examples:
  python generate_test_ids.py --mode all
  python generate_test_ids.py --mode first --n 117
  python generate_test_ids.py --mode random --n 117 --seed 42
"""
import argparse
import os
import shutil
import sys
import random

def parse_args():
    p = argparse.ArgumentParser(description="Generate test IDs file from dataset (BraTS project)")
    p.add_argument("--mode", choices=["all", "first", "random"], default="all",
                   help="How to build the test set: all / first / random (default: all)")
    p.add_argument("--n", type=int, default=117,
                   help="Number of test IDs when mode is first or random (default 117)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling (default 42)")
    p.add_argument("--out", default="test_ids.txt",
                   help="Output filename (default: test_ids.txt)")
    p.add_argument("--no-backup", action="store_true",
                   help="Do not create a backup of existing output file")
    return p.parse_args()

def try_import_get_datasets():
    """
    Try to import get_datasets from brats. Handle cases where brats.get_datasets
    is either a callable function OR an object/instance/class named get_datasets.
    Return a callable that yields the dataset (list-like).
    """
    try:
        import brats
        # common case: brats.get_datasets is a function
        if hasattr(brats, "get_datasets"):
            gd = brats.get_datasets
            if callable(gd):
                return gd
            # gd exists but not callable (e.g., an already-instantiated object or class)
            # return a wrapper that returns it (or an instance)
            try:
                # If it's a class, instantiate it
                if isinstance(gd, type):
                    inst = gd()
                    return lambda: inst
            except Exception:
                pass
            # If it's an object/instance, return it directly via a wrapper
            return lambda: gd

        # fallback: maybe there's a class named Brats
        if hasattr(brats, "Brats"):
            Brats = brats.Brats
            if callable(Brats):
                return lambda: Brats()

        raise ImportError("no get_datasets or Brats found in brats module")
    except Exception as e:
        print("ERROR: failed to import dataset access from 'brats' module.", file=sys.stderr)
        print("Reason:", repr(e), file=sys.stderr)
        print("Run from project root or set PYTHONPATH so Python can import `brats`:")
        print("  PYTHONPATH=. python generate_test_ids.py", file=sys.stderr)
        sys.exit(2)

def collect_ids(get_datasets_callable):
    """
    get_datasets_callable may be:
    - a function that returns a dataset (call it)
    - a wrapper that returns a dataset-like object
    - or the dataset object itself (in which case it will be returned)
    """
    ds = get_datasets_callable()
    ids = []
    for i in range(len(ds)):
        item = ds[i]
        pid = item.get("patient_id", None) or item.get("id", None) or item.get("name", None) or str(i)
        ids.append(str(pid))
    return ids

def main():
    args = parse_args()
    get_datasets = try_import_get_datasets()
    ids = collect_ids(get_datasets)
    total = len(ids)
    if total == 0:
        print("ERROR: dataset appears empty.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out

    # backup existing file if present
    if os.path.exists(out_path) and not args.no_backup:
        bak = out_path + ".bak"
        shutil.copy2(out_path, bak)
        print(f"Existing '{out_path}' backed up as '{bak}'")

    if args.mode == "all":
        selected = ids
        print(f"Selecting ALL {total} IDs")
    elif args.mode == "first":
        if args.n > total:
            print(f"Requested n={args.n} > total {total}. Using all IDs instead.")
            selected = ids
        else:
            selected = ids[:args.n]
        print(f"Selecting FIRST {len(selected)} IDs")
    elif args.mode == "random":
        if args.n > total:
            print(f"Requested n={args.n} > total {total}. Using all IDs instead.")
            selected = ids
        else:
            random.seed(args.seed)
            selected = random.sample(ids, args.n)
        print(f"Selecting RANDOM {len(selected)} IDs (seed={args.seed})")
    else:
        raise RuntimeError("Unknown mode")

    # write file
    with open(out_path, "w", encoding="utf-8") as f:
        for pid in selected:
            f.write(pid + "\n")

    print(f"Wrote {len(selected)} IDs to '{out_path}'. Sample (first 10):")
    for x in selected[:10]:
        print("  ", x)

    print("\nIMPORTANT:")
    print(" - If your model was trained using any of these patients, evaluating on them may cause data leakage.")
    print(" - If you have saved original train/val/test split used during training, use that exact test_ids file instead.")
    print(" - Use --no-backup to skip backing up an existing output file.")

if __name__ == "__main__":
    main()