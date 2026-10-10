import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Discrete Wavelet Transform (DWT) Layer ---
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return self.dwt_init(x)

    def dwt_init(self, x):
        x01 = x[:, :, :, 0::2, :] / 2
        x02 = x[:, :, :, 1::2, :] / 2
        x1 = x01[:, :, :, :, 0::2]
        x2 = x02[:, :, :, :, 0::2]
        x3 = x01[:, :, :, :, 1::2]
        x4 = x02[:, :, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4
        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)

# --- 1b. 3D Haar DWT (all three spatial axes) ---
class DWT3D(nn.Module):
    """Orthonormal single-level 3D Haar transform: [B,C,D,H,W] -> [B,8C,D/2,H/2,W/2].

    The 2D `DWT` above halves only the last two axes, so the slice axis is never
    downsampled and the encoder's receptive field along it stays small. This
    halves all three, emitting the 8 sub-bands (LLL ... HHH). Orthonormal
    (scaled by 1/(2*sqrt(2))), so it is exactly invertible and energy-preserving:
    no information is discarded by the downsampling step.
    """

    def forward(self, x):
        s = 2.0 ** -1.5
        a, b = x[:, :, 0::2], x[:, :, 1::2]          # depth
        bands = []
        for d in (a + b, b - a):
            c, e = d[:, :, :, 0::2], d[:, :, :, 1::2]   # height
            for h in (c + e, e - c):
                f, g = h[..., 0::2], h[..., 1::2]       # width
                bands += [f + g, g - f]
        return torch.cat(bands, dim=1) * s

# --- 2. Convolution Block ---
def _norm(kind, ch):
    """'batch' = the published model. 'instance' normalises each volume by its own
    statistics (as nnU-Net does): no running statistics carried over from training,
    which matters at batch size 2 and under scanner shift."""
    if kind == "batch":
        return nn.BatchNorm3d(ch)
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    raise ValueError(f"unknown norm {kind!r}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="batch"):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            _norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            _norm(norm, out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- 3. Wavelet U-Net++ Architecture ---
class WaveletUNetPlusPlus(nn.Module):
    """U-Net++ backbone whose encoder downsampling operator is selectable.

    ``downsample="dwt"`` is the proposed model and the default; it reproduces
    the original module layout exactly, so existing checkpoints load unchanged.

    ``downsample="maxpool_matched"`` is the ablation baseline (paper Table IX,
    variant A). Max-pooling halves the same two in-plane axes the Haar DWT does,
    but emits C channels where the DWT emits 4C. A 1x1 projection restores that
    width so every encoder block receives an identically shaped tensor under
    both arms. Without the projection the baseline would carry 34% fewer
    parameters (6.87M vs 10.40M) and any gap would confound model capacity with
    the frequency content the wavelet claim is actually about; with it the two
    arms sit at 10.48M vs 10.40M, a 0.8% difference.
    """

    def __init__(self, in_channels=4, n_classes=3, downsample="dwt",
                 deep_supervision=False, base_filters=16, norm="batch"):
        """`deep_supervision` attaches auxiliary 1x1 heads to the three shallower
        nested decoder outputs (x0_1, x0_2, x0_3), which U-Net++ already produces
        at full resolution.

        Off by default and adds no parameters when off, so published checkpoints
        still load with strict=True. When on, `self.final` is unchanged and is
        still the only head used at inference -- the auxiliary logits are
        returned during training only, so evaluation code needs no change.
        """
        super(WaveletUNetPlusPlus, self).__init__()
        # base_filters=16 is the published width; larger values scale every stage.
        nb_filter = [base_filters * m for m in (1, 2, 4, 8, 16)]
        self.deep_supervision = deep_supervision
        if downsample not in ("dwt", "maxpool_matched", "dwt3d", "maxpool3d_matched"):
            raise ValueError(
                "downsample must be 'dwt', 'dwt3d', 'maxpool_matched' or 'maxpool3d_matched', got %r" % (downsample,))
        self.downsample = downsample
        # Sub-band count: the 2D DWT emits 4C channels, the 3D DWT 8C.
        k = 8 if downsample in ("dwt3d", "maxpool3d_matched") else 4
        self.dwt = DWT() if downsample == "dwt" else (DWT3D() if downsample == "dwt3d" else None)
        self.expand = None if downsample in ("dwt", "dwt3d") else nn.ModuleList(
            [nn.Conv3d(c, c * k, kernel_size=1) for c in nb_filter[:4]])
        # --- Encoders ---
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], norm=norm)
        self.conv1_0 = ConvBlock(nb_filter[0]*k, nb_filter[1], norm=norm)
        self.conv2_0 = ConvBlock(nb_filter[1]*k, nb_filter[2], norm=norm)
        self.conv3_0 = ConvBlock(nb_filter[2]*k, nb_filter[3], norm=norm)
        self.conv4_0 = ConvBlock(nb_filter[3]*k, nb_filter[4], norm=norm)
        # --- Decoders ---
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], norm=norm)
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], norm=norm)
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], norm=norm)
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], norm=norm)
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], norm=norm)
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], norm=norm)
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], norm=norm)
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], norm=norm)
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], norm=norm)
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], norm=norm)
        self.up = nn.Upsample(scale_factor=(2, 2, 2) if downsample in ("dwt3d", "maxpool3d_matched") else (1, 2, 2),
                              mode='trilinear', align_corners=True)
        self.final = nn.Conv3d(nb_filter[0], n_classes, kernel_size=1)
        # Created only when deep supervision is on, so the default state_dict
        # stays identical to the published model's.
        self.ds_heads = nn.ModuleList(
            [nn.Conv3d(nb_filter[0], n_classes, kernel_size=1) for _ in range(3)]
        ) if deep_supervision else None

    def _down(self, x, level):
        """Halve the two in-plane axes (4C channels), or all three for dwt3d (8C)."""
        if self.downsample in ("dwt", "dwt3d"):
            return self.dwt(x)
        if self.downsample == "maxpool3d_matched":
            # Non-wavelet control for the 3D DWT (PROTOCOL_v2 Amendment 21): 2x2x2 max-pooling,
            # then a 1x1x1 projection C -> 8C so every encoder block sees the same shape as with dwt3d.
            pooled = F.max_pool3d(x, kernel_size=2, stride=2)
        else:
            # Depth is kept at full resolution to match the DWT, hence (1, 2, 2).
            pooled = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return self.expand[level](pooled)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self._down(x0_0, 0))
        x2_0 = self.conv2_0(self._down(x1_0, 1))
        x3_0 = self.conv3_0(self._down(x2_0, 2))
        x4_0 = self.conv4_0(self._down(x3_0, 3))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        out = self.final(x0_4)
        if self.ds_heads is not None and self.training:
            # Shallow-to-deep; the caller weights them. Training only, so
            # inference and evaluation see exactly the single-head model.
            aux = [h(x) for h, x in zip(self.ds_heads, (x0_1, x0_2, x0_3))]
            return aux + [out]
        return out