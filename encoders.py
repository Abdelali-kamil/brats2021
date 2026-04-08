import torch
import torch.nn as nn

# --- 1. Basic 3D CNN Encoder Block ---
# This block extracts features from 3D volumes (like ResNet blocks but simpler)
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_features):
        super(EncoderBlock, self).__init__()
        
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # Downsample
            
            # Layer 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # Downsample
            
            # Layer 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Global Pooling to flatten spatial dimensions into a single vector
            nn.AdaptiveAvgPool3d((1, 1, 1)) 
        )
        
        # Fully Connected Layer to get the desired feature size
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1) # Flatten to [Batch, 128]
        return self.fc(features)

# --- 2. Multi-Scale Feature Extractor (Part 2 of Diagram) ---
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, feature_dim=512):
        super(FeatureExtractor, self).__init__()
        
        # Lesion Encoder: Focuses only on the tumor area
        self.lesion_encoder = EncoderBlock(in_channels, feature_dim // 2)
        
        # Global/Slice Encoder: Looks at the entire image for context
        self.global_encoder = EncoderBlock(in_channels, feature_dim // 2)
        
    def forward(self, image, mask):
        """
        Args:
            image: Original 3D Image [Batch, 4, D, H, W]
            mask: Binary Mask from Part 1 [Batch, 1, D, H, W]
        Returns:
            Combined Feature Vector [Batch, feature_dim]
        """
        
        # 1. Prepare Input for Lesion Encoder
        # We multiply image by mask to zero out the background, keeping only the tumor.
        # Note: Mask is [Batch, 1, ...], Image is [Batch, 4, ...]. Broadcasting works automatically.
        masked_img = image * mask 
        
        # 2. Extract Features
        lesion_features = self.lesion_encoder(masked_img) # Local Features
        global_features = self.global_encoder(image)      # Global Context Features
        
        # 3. Patient-Level Feature Fusion
        # Concatenate both vectors (e.g., 256 + 256 = 512)
        combined_features = torch.cat([lesion_features, global_features], dim=1)
        
        return combined_features

# --- Test Script ---
if __name__ == "__main__":
    # Simulate Data
    # Image: Batch=1, Channels=4, Depth=32, H=128, W=128
    image = torch.randn(1, 4, 32, 128, 128) 
    
    # Mask: Random binary mask (0 or 1)
    mask = torch.randint(0, 2, (1, 1, 32, 128, 128)).float() 
    
    model = FeatureExtractor(in_channels=4, feature_dim=512)
    
    print("⏳ Testing Part 2: Feature Extractor...")
    output = model(image, mask)
    
    print(f"✅ Part 2 Successful!")
    print(f"   Combined Features Shape: {output.shape}") 
    # Expected Output: [1, 512]