#!/usr/bin/env python3
"""
compare_ids.py

Compare test_ids.txt vs dataset patient IDs and print diagnostics.
Run: python compare_ids.py
"""
import os, re, sys
from brats import get_datasets

TEST = "test_ids.txt"

def norm(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'\.nii(?:\.gz)?$', '', s)
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s

# load dataset IDs
ds = get_datasets()
dataset_raw = []
for i in range(len(ds)):
    item = ds[i]
    pid = item.get("patient_id", None) or item.get("id", None) or str(i)
    dataset_raw.append(str(pid))

dataset_norm = [norm(x) for x in dataset_raw]
dataset_set_raw = set(dataset_raw)
dataset_set_norm = set(dataset_norm)

print("Dataset size:", len(dataset_raw))
print("\nFirst 30 raw dataset IDs:")
for x in dataset_raw[:30]:
    print("  ", x)

print("\nFirst 30 normalized dataset IDs:")
for x in dataset_norm[:30]:
    print("  ", x)

# load test_ids file
if not os.path.exists(TEST):
    print(f"\nERROR: {TEST} not found in cwd.")
    sys.exit(1)

with open(TEST,'r',encoding='utf-8') as f:
    test_lines = [ln.rstrip("\n\r") for ln in f if ln.strip()]

print(f"\nLoaded {len(test_lines)} entries from {TEST}")
print("\nFirst 30 lines from test_ids.txt:")
for x in test_lines[:30]:
    print("  ", x)

# normalized forms
test_norm = [norm(x) for x in test_lines]
test_set_norm = set(test_norm)

# intersection stats
direct_raw_matches = sum(1 for t in test_lines if t in dataset_set_raw)
norm_matches = sum(1 for tn in test_norm if tn in dataset_set_norm)
print("\nDirect raw string matches:", direct_raw_matches)
print("Normalized-string matches:", norm_matches)

# print some unmatched test entries (normalized form and candidate matches)
unmatched = [t for t,tn in zip(test_lines,test_norm) if tn not in dataset_set_norm and t not in dataset_set_raw]
print(f"\nUnmatched test entries (show up to 40): {len(unmatched)}")
for u in unmatched[:40]:
    print("  ", u, "  -> normalized:", norm(u))

# show dataset IDs not requested in test list (sample)
missing_from_test = [d for d,n in zip(dataset_raw,dataset_norm) if n not in test_set_norm and d not in test_lines]
print(f"\nNumber of dataset IDs not present in test_ids.txt: {len(missing_from_test)} (show first 20)")
for x in missing_from_test[:20]:
    print("  ", x)

# show numeric-only test IDs and try numeric match to dataset
def numeric(s):
    m = re.search(r'(\d{3,})', str(s))
    return m.group(1) if m else None

numeric_test = [(t, numeric(t)) for t in test_lines if numeric(t)]
print(f"\nNumeric-only or numeric-containing test lines (count {len(numeric_test)}), sample:")
for t,n in numeric_test[:20]:
    candidates = [d for d in dataset_raw if numeric(d) == n]
    print("  ", t, "=>", n, "candidates:", candidates[:5])

print("\nDone. Paste the printed output here and I'll provide the exact corrected test_ids.txt or the next edit to analyze_results.py.")