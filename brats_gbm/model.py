import torch
import torch.nn as nn

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
def _make_norm(norm: str, num_features: int) -> nn.Module:
    """Normalisation layer factory.

    'batch' is the original behaviour. 'instance' and 'group' do not depend on
    batch statistics, which matters here because training runs at an effective
    batch size of ~1 (BatchNorm running statistics are then both noisy and
    specific to the BraTS intensity distribution, hurting zero-shot transfer to
    UPenn). Instance/Group norm are the nnU-Net-standard choice for that reason.
    """
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm3d(num_features)
    if norm == "instance":
        return nn.InstanceNorm3d(num_features, affine=True)
    if norm == "group":
        # 8 groups (or fewer if channels < 8), a common default.
        groups = min(8, num_features)
        while num_features % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_features)
    raise ValueError(f"unknown norm {norm!r}; expected batch|instance|group")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="batch"):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            _make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            _make_norm(norm, out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- 3. Wavelet U-Net++ Architecture ---
class WaveletUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=4, n_classes=3, norm="batch"):
        super(WaveletUNetPlusPlus, self).__init__()
        nb_filter = [16, 32, 64, 128, 256]
        self.dwt = DWT()
        # --- Encoders ---
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], norm=norm)
        self.conv1_0 = ConvBlock(nb_filter[0]*4, nb_filter[1], norm=norm)
        self.conv2_0 = ConvBlock(nb_filter[1]*4, nb_filter[2], norm=norm)
        self.conv3_0 = ConvBlock(nb_filter[2]*4, nb_filter[3], norm=norm)
        self.conv4_0 = ConvBlock(nb_filter[3]*4, nb_filter[4], norm=norm)
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
        
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.final = nn.Conv3d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.dwt(x0_0))
        x2_0 = self.conv2_0(self.dwt(x1_0))
        x3_0 = self.conv3_0(self.dwt(x2_0))
        x4_0 = self.conv4_0(self.dwt(x3_0))

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