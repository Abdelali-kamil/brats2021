#!/usr/bin/env python3
"""Average saved sigmoid probability maps (<case>.npz, key 'prob') across
member directories into OUT. Every member must hold the same case set."""
import sys, pathlib, numpy as np
out = pathlib.Path(sys.argv[1]); members = [pathlib.Path(d) for d in sys.argv[2:]]
assert len(members) >= 2, "need at least two members"
sets = [sorted(p.name for p in m.glob("*.npz")) for m in members]
assert all(s == sets[0] for s in sets) and sets[0], "members differ in cases"
out.mkdir(parents=True, exist_ok=True)
for name in sets[0]:
    acc = sum(np.load(m / name)["prob"].astype(np.float32) for m in members) / len(members)
    np.savez_compressed(out / name, prob=acc.astype(np.float16))
print(f"averaged {len(sets[0])} cases from {len(members)} members -> {out}")
