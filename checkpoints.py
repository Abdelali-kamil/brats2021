import torch
from model import WaveletUNetPlusPlus

device = "cuda" if torch.cuda.is_available() else "cpu"
paths = [
    "checkpoints/segmentor_epoch_750.pth"
]

for p in paths:
    print(f"\nTesting: {p}")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    ckpt = torch.load(p, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    try:
        model.load_state_dict(sd, strict=True)
        print("✅ Compatible (strict=True)")
    except Exception as e:
        print("❌ Not compatible:", e)