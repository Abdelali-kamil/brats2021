"""Shared helpers for the efficiency benchmark (PROTOCOL_v2 'Efficiency').
Timing = in-memory raw 4-channel volume -> final segmentation, under each
method's OWN inference protocol; disk I/O excluded; CUDA-synchronised.
1 warm-up case (not recorded) + 10 timed test cases, identical for all methods."""
import json, time, statistics, pathlib
import torch
DATA = "/mnt/data1/kamil_research/data/brats2021"
IDS = [l.strip() for l in open("/home/kamilabdelali/brats2021/baselines/common/test_ids.txt") if l.strip()]
import os
SMOKE = os.environ.get("BENCH_SMOKE") == "1"   # functional check only: 2 cases, separate dir
WARMUP, CASES = IDS[0], (IDS[1:3] if SMOKE else IDS[1:11])
OUT = pathlib.Path("/mnt/data1/kamil_research/experiments/efficiency/" + ("smoke" if SMOKE else "results"))

class Counter:
    """Counts top-level forward calls (= network evaluations per case)."""
    def __init__(self, module): self.n = 0; module.register_forward_hook(self._hook)
    def _hook(self, *a): self.n += 1

def flops_per_forward(model, shape, device="cuda"):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(shape, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()

def timed(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    out = fn(); torch.cuda.synchronize()
    return out, time.perf_counter() - t, torch.cuda.max_memory_allocated() / 2**30

def save(name, params, flops, fwd_shape, rows, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    ts = [r["sec"] for r in rows]; mem = [r["peak_gib"] for r in rows]; nf = [r["n_forward"] for r in rows]
    res = {"method": name, "params": params, "params_M": round(params / 1e6, 2),
           "gflops_per_forward": round(flops / 1e9, 1), "forward_input_shape": list(fwd_shape),
           "forwards_per_case_mean": statistics.mean(nf),
           "sec_per_case_mean": round(statistics.mean(ts), 2), "sec_per_case_sd": round(statistics.stdev(ts), 2),
           "peak_gpu_gib_max": round(max(mem), 2), "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cases": CASES, "per_case": rows, **(extra or {})}
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_case", "cases")}, indent=1))
