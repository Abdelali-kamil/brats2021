"""Amendment 25 step 8: Henry et al. (EquiUnet w48, deep supervision, final SWA model) under the inference it was scored
with (src/inference.py --tta, patched 2026-10-12): min-max per modality, crop to the non-zero box (+1 voxel), pad to a
multiple of 16, simple TTA (identity + 15 flip/rotation transforms, sigmoid, mean) under fp16 autocast, un-pad, un-crop
into 155x240x240, threshold 0.5, its label assembly. --no-tta: one forward pass."""
import sys, argparse, glob, numpy as np, torch, yaml, SimpleITK as sitk
H = "/mnt/data1/kamil_research/experiments/henry_dsunet"
sys.path.insert(0, f"{H}/code"); sys.path.insert(0, f"{H}/ranger_pkg"); sys.path.insert(0, "/mnt/data1/kamil_research/experiments/efficiency")
from common import *
from src import models
from src.models import get_norm_layer
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.dataset.image_utils import irm_min_max_preprocess
from src.tta import apply_simple_tta
from src.utils import reload_ckpt_bis
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="henry_dsunet"); ap.add_argument("--no-tta", action="store_true"); a = ap.parse_args()
RUN = open(f"{H}/RUN_PATH").read().strip(); cfg = yaml.safe_load(open(glob.glob(f"{RUN}/*.yaml")[0]))
m = getattr(models, cfg["arch"])(4, 3, width=cfg["width"], deep_supervision=cfg["deep_sup"], norm_layer=get_norm_layer(cfg["norm_layer"]), dropout=cfg["dropout"])
reload_ckpt_bis(f"{RUN}/model_best.pth.tar", m); m = m.cuda().eval(); cnt = Counter(m)
def load(c):
    return np.stack([sitk.GetArrayFromImage(sitk.ReadImage(f"{DATA}/{c}/{c}_{s}.nii.gz")) for s in ("t1", "t1ce", "t2", "flair")])
def run(raw):
    img = np.stack([irm_min_max_preprocess(ch) for ch in raw])
    z, y, x = np.nonzero(np.sum(img, axis=0) != 0)
    lo = [max(0, int(np.min(v) - 1)) for v in (z, y, x)]; hi = [int(np.max(v) + 1) for v in (z, y, x)]
    t = torch.from_numpy(img[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype("float16"))[None]
    t, pads = pad_batch1_to_compatible_size(t); t = t.cuda()
    with torch.autocast("cuda"):
        if a.no_tta:
            p = m(t)[0].sigmoid_().cpu() if cfg["deep_sup"] else m(t).sigmoid_().cpu()
        else:
            p = apply_simple_tta(m, t, True)
    p = p[:, :, :p.size(2) - pads[0], :p.size(3) - pads[1], :p.size(4) - pads[2]]
    segs = torch.zeros((1, 3, 155, 240, 240)); segs[0, :, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = p[0].float()
    s = segs[0].numpy() > 0.5; et = s[0]; net = np.logical_and(s[1], ~et); ed = np.logical_and(s[2], ~s[1])
    lab = np.zeros(et.shape, np.uint8); lab[et] = 4; lab[net] = 1; lab[ed] = 2
    return lab
with torch.no_grad():
    run(load(WARMUP)); rows = []
    for c in CASES:
        raw = load(c); cnt.n = 0
        _, s, g = timed(lambda: run(raw)); rows.append({"case": c, "sec": s, "peak_gib": g, "n_forward": cnt.n})
save(a.name, sum(p.numel() for p in m.parameters()), flops_per_forward(m, (1, 4, 128, 128, 128)), (1, 4, 128, 128, 128), rows,
     {"protocol": ("no TTA" if a.no_tta else "simple TTA (16 transforms)") + ", fp16 autocast, whole cropped volume per pass; GFLOPs at 128^3",
      "note": "passes are whole brain-cropped volumes, not fixed windows"})
