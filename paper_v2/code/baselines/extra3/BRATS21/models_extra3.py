"""Amendment 19: networks for the three extra baselines, built from the authors' official implementations.
  unet3d    : pytorch-3dunet UNet3D (wolny/pytorch-3dunet @ a33e2c7), its default configuration
  segresnet : MONAI SegResNet, configuration of the official MONAI BraTS tutorial (brats_segmentation_3d.ipynb @ b4b61f8)
  unetr     : MONAI UNETR, configuration of the official UNETR code (research-contributions/UNETR/BTCV @ 21ed8e5)
  swinunetr : MONAI SwinUNETR as in the existing Swin UNETR baseline
All return raw logits (sigmoid is applied by the pipeline)."""
import sys
sys.path.insert(0, "/home/kamilabdelali/brats2021/baselines/repos/pytorch_3dunet")
ROI = {"unet3d": (96, 96, 96), "unetr": (96, 96, 96), "segresnet": (224, 224, 144), "swinunetr": (96, 96, 96)}

def build_model(name, in_channels=4, out_channels=3, roi=None):
    roi = tuple(roi or ROI[name])
    if name == "unet3d":
        from pytorch3dunet.unet3d.model import UNet3D
        return UNet3D(in_channels=in_channels, out_channels=out_channels, final_sigmoid=True, is_segmentation=False)
    if name == "segresnet":
        from monai.networks.nets import SegResNet
        return SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
                         in_channels=in_channels, out_channels=out_channels, dropout_prob=0.2)
    if name == "unetr":
        from monai.networks.nets import UNETR
        return UNETR(in_channels=in_channels, out_channels=out_channels, img_size=roi, feature_size=16, hidden_size=768,
                     mlp_dim=3072, num_heads=12, proj_type="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)
    if name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(in_channels=in_channels, out_channels=out_channels, feature_size=48)
    raise ValueError(name)
