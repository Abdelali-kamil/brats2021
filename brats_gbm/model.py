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

# --- 2. Convolution Block ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
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

    def __init__(self, in_channels=4, n_classes=3, downsample="dwt"):
        super(WaveletUNetPlusPlus, self).__init__()
        nb_filter = [16, 32, 64, 128, 256]
        if downsample not in ("dwt", "maxpool_matched"):
            raise ValueError(
                "downsample must be 'dwt' or 'maxpool_matched', got %r" % (downsample,))
        self.downsample = downsample
        self.dwt = DWT() if downsample == "dwt" else None
        self.expand = None if downsample == "dwt" else nn.ModuleList(
            [nn.Conv3d(c, c * 4, kernel_size=1) for c in nb_filter[:4]])
        # --- Encoders ---
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0]*4, nb_filter[1]) 
        self.conv2_0 = ConvBlock(nb_filter[1]*4, nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2]*4, nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3]*4, nb_filter[4])
        # --- Decoders ---
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])
        
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.final = nn.Conv3d(nb_filter[0], n_classes, kernel_size=1)

    def _down(self, x, level):
        """Halve the two in-plane axes, emitting 4C channels either way."""
        if self.downsample == "dwt":
            return self.dwt(x)
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
        return self.final(x0_4)