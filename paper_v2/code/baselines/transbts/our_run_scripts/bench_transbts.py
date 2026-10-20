"""Amendment 24/25 step 8: TransBTS under the inference it is scored with (test.py/predict.py): z-score over the brain mask
(preprocess.py process_f32b0), pad 155->160, 8 overlapping 128^3 crops (tailor_and_concat), 8-flip TTA with softmax mean
(--no-tta: one pass of the 8 crops), argmax, labels 1/2/4. Weights = final (last-epoch) model."""
import sys, argparse, numpy as np, nibabel as nib, torch, torch.nn.functional as F
W = "/mnt/data1/kamil_research/experiments/transbts"
sys.path.insert(0, f"{W}/code"); sys.path.insert(0, "/mnt/data1/kamil_research/experiments/efficiency")
from common import *
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from predict import tailor_and_concat
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="transbts"); ap.add_argument("--no-tta", action="store_true")
ap.add_argument("--ckpt", default=f"{W}/code/checkpoint/TransBTS_fixedsplit_20261014/model_epoch_last.pth"); a = ap.parse_args()
_, net = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
m = torch.nn.DataParallel(net).cuda(); m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]); m.eval()
cnt = Counter(net)
def load(c):
    return np.stack([np.asarray(nib.load(f"{DATA}/{c}/{c}_{s}.nii.gz").dataobj, dtype=np.float32) for s in ("flair", "t1ce", "t1", "t2")], -1)
def run(raw):
    img = raw.copy(); mask = img.sum(-1) > 0
    for k in range(4):
        x = img[..., k]; y = x[mask]; x[mask] -= y.mean(); x[mask] /= y.std(); img[..., k] = x
    img = np.pad(img, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(3, 0, 1, 2)))[None].float()
    if a.no_tta:
        out = F.softmax(tailor_and_concat(x, m), 1)
    else:
        x = x[..., :155]; out = F.softmax(tailor_and_concat(x, m), 1)
        for d in ((2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)):
            out += F.softmax(tailor_and_concat(x.flip(dims=d), m).flip(dims=d), 1)
        out = out / 8.0
    o = out[0, :, :240, :240, :155].cpu().numpy().argmax(0)
    lab = np.zeros(o.shape, np.uint8); lab[o == 1] = 1; lab[o == 2] = 2; lab[o == 3] = 4
    return lab
with torch.no_grad():
    run(load(WARMUP)); rows = []
    for c in CASES:
        raw = load(c); cnt.n = 0
        _, s, g = timed(lambda: run(raw)); rows.append({"case": c, "sec": s, "peak_gib": g, "n_forward": cnt.n})
save(a.name, sum(p.numel() for p in net.parameters()), flops_per_forward(net, (1, 4, 128, 128, 128)), (1, 4, 128, 128, 128), rows,
     {"protocol": "8 crops 128^3, " + ("no TTA" if a.no_tta else "8-flip TTA") + ", softmax mean, argmax, fp32 (as scored)"})
