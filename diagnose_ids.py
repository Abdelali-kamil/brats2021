import os
import re
import sys
from brats import get_datasets

def norm(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'\.nii(?:\.gz)?$', '', s)
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s

ds = get_datasets()
print(f"📂 Found {len(ds)} candidate patient folders.")
print("🔍 Scan initiated: Checking {} folders...".format(len(ds)))

dataset_raw = []
for i in range(len(ds)):
    item = ds[i]
    pid = item.get("patient_id", None) or item.get("id", None) or str(i)
    dataset_raw.append(str(pid))

dataset_norm = [norm(x) for x in dataset_raw]

print("✅ Success: {} patients loaded from {} folders.".format(len(dataset_raw), len(dataset_raw)))
print("Dataset size:", len(dataset_raw))
print("\nSample of normalized dataset IDs (first 40):")
for x in dataset_norm[:40]:
    print("  ", x)
print("-" * 60)
print("\nSample of raw dataset IDs (first 40):")
for x in dataset_raw[:40]:
    print("  ", x)
print("-" * 60)

TEST = "test_ids.txt"
if not os.path.exists(TEST):
    print(f"\nERROR: {TEST} not found in cwd.")
    sys.exit(1)

with open(TEST, "r", encoding="utf-8") as f:
    test_lines = [ln.rstrip("\n\r") for ln in f if ln.strip()]

print(f"\nLoaded {len(test_lines)} test IDs from: {TEST}")
print("Sample test IDs (first 40):")
for x in test_lines[:40]:
    print("  ", x)
print("-" * 60)

# quick overlap stats
test_norm = [norm(x) for x in test_lines]
overlap = set(dataset_norm).intersection(set(test_norm))
print(f"Direct normalized overlap count: {len(overlap)}")
if len(overlap) > 0:
    print("Example overlaps (up to 20):")
    for x in list(overlap)[:20]:
        print("  ", x)
print("-" * 60)

# try simple heuristic matching
def numeric(s):
    m = re.search(r'(\d{3,})', str(s))
    return m.group(1) if m else None

matches = {}
unmatched = []
for t in test_lines:
    tn = norm(t)
    # direct normalized
    if tn in dataset_norm:
        matches[t] = tn
        continue
    # numeric match
    tn_num = numeric(t)
    if tn_num:
        for d in dataset_raw:
            if numeric(d) == tn_num:
                matches[t] = d
                break
    if t not in matches:
        unmatched.append(t)

print(f"Matches found: {len(matches)}  |  Unmatched: {len(unmatched)}")
if matches:
    print("\nSample matched pairs (test_id => dataset_norm_or_raw):")
    shown = 0
    for t,d in matches.items():
        print("  ", t, "=>", d)
        shown += 1
        if shown >= 30:
            break

if unmatched:
    print("\nSample unmatched test IDs (up to 30):")
    for x in unmatched[:30]:
        print("  ", x)
print("\nDone.")