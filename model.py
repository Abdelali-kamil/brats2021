import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Discrete Wavelet Transform (DWT) Layer ---
# This replaces standard pooling. It downsamples the image while keeping frequency information.
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # DWT is fixed, not trained

    def forward(self, x):
        return self.dwt_init(x)

    def dwt_init(self, x):
        # We apply Haar Wavelet transform on Height (H) and Width (W) dimensions
        # x shape: [batch, channel, depth, height, width]
        
        x01 = x[:, :, :, 0::2, :] / 2
        x02 = x[:, :, :, 1::2, :] / 2
        
        x1 = x01[:, :, :, :, 0::2]
        x2 = x02[:, :, :, :, 0::2]
        x3 = x01[:, :, :, :, 1::2]
        x4 = x02[:, :, :, :, 1::2]
        
        # The 4 sub-bands: LL (approx), LH, HL, HH (details)
        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4
        
        # Concatenate results: The channel count increases by 4x
        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)

# --- 2. Convolution Block ---
# Standard double convolution layer with Batch Normalization and ReLU
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
    def __init__(self, in_channels=4, n_classes=3):
        super(WaveletUNetPlusPlus, self).__init__()
        
        # Number of filters at each level
        nb_filter = [16, 32, 64, 128, 256]

        self.dwt = DWT() # Initialize Wavelet Layer

        # --- Encoder (Wavelet Path) ---
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0]*4, nb_filter[1]) # Input channels * 4 because of DWT
        self.conv2_0 = ConvBlock(nb_filter[1]*4, nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2]*4, nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3]*4, nb_filter[4])

        # --- Decoders (Nested U-Net++ Paths) ---
        # Level 1
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        # Level 2
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        # Level 3
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        # Level 4 (Final Output Path)
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # Upsampling (We resize H and W only, matching DWT downsampling)
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        
        # Final Segmentation Head
        self.final = nn.Conv3d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        # --- Encoder Steps ---
        x0_0 = self.conv0_0(input)
        
        w1 = self.dwt(x0_0) # Downsample 1
        x1_0 = self.conv1_0(w1)
        
        w2 = self.dwt(x1_0) # Downsample 2
        x2_0 = self.conv2_0(w2)
        
        w3 = self.dwt(x2_0) # Downsample 3
        x3_0 = self.conv3_0(w3)
        
        w4 = self.dwt(x3_0) # Downsample 4
        x4_0 = self.conv4_0(w4)

        # --- Decoder Steps (Dense Skip Connections) ---
        # L1 Nodes
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # L2 Nodes
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # L3 Nodes
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # L4 Node (Output)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

# --- Test Script ---
if __name__ == "__main__":
    # Create a random 3D image (Batch=1, Channels=4, Depth=32, Height=128, Width=128)
    # Note: Depth is kept smaller here (32) to save memory during test
    image = torch.randn(1, 4, 32, 128, 128)
    
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3)
    
    print("⏳ Running model test...")
    output = model(image)
    
    print(f"✅ Model Built Successfully!")
    print(f"   Input Shape: {image.shape}")
    print(f"   Output Shape: {output.shape}")