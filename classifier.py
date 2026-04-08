import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. KAN Layer (Fixed) ---
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Base weight (linear transformation)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline weights (non-linear transformation)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.constant_(self.spline_weight, 0.1)

    def forward(self, x):
        # x: [Batch, In_Features]
        
        # 1. Base Linear Path
        base_output = F.linear(x, self.base_weight)
        
        # 2. Non-linear Spline Path (Corrected)
        # Apply activation -> Then Linear Projection
        spline_output = F.linear(F.silu(x), self.spline_weight)

        return base_output + spline_output

# --- 2. GNN-Enhanced Memory Bank ---
class GNNMemoryBank(nn.Module):
    def __init__(self, feature_dim, memory_size=100):
        super(GNNMemoryBank, self).__init__()
        # Memory slots
        self.memory = nn.Parameter(torch.randn(1, memory_size, feature_dim))
        
        # Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x: [Batch, Feature_Dim] -> [Batch, 1, Feature_Dim]
        x = x.unsqueeze(1) 
        
        # Expand memory to match batch size
        batch_size = x.size(0)
        mem = self.memory.expand(batch_size, -1, -1)
        
        # Graph Attention
        attn_output, _ = self.attention(x, mem, mem)
        
        return self.norm(x + attn_output).squeeze(1)

# --- 3. The Main Classifier Architecture ---
class AdvancedClassifier(nn.Module):
    def __init__(self, feature_dim=512, clinical_dim=10, num_classes=2):
        super(AdvancedClassifier, self).__init__()
        
        # A. Clinical Data Encoder
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # B. Transformer Branch
        self.transformer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        
        # C. KAN Branch
        self.kan_layer = KANLinear(feature_dim, feature_dim)
        
        # D. GNN Memory
        self.memory_bank = GNNMemoryBank(feature_dim)
        
        # E. Fusion Gate
        self.fusion_dim = feature_dim + 128
        self.gate = nn.Linear(self.fusion_dim, self.fusion_dim)
        
        # F. Final Classifier
        self.final_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_features, clinical_data):
        # 1. Clinical
        clin_feat = self.clinical_net(clinical_data) 
        
        # 2. Transformer
        trans_feat = self.transformer(img_features.unsqueeze(1)).squeeze(1)
        
        # 3. KAN
        kan_feat = self.kan_layer(img_features)
        
        # 4. Memory Refinement
        combined_img = trans_feat + kan_feat
        mem_feat = self.memory_bank(combined_img)
        
        # 5. Fusion
        fusion_input = torch.cat([mem_feat, clin_feat], dim=1) 
        
        # Gating
        gate_val = torch.sigmoid(self.gate(fusion_input))
        fusion_output = fusion_input * gate_val
        
        # 6. Logits
        logits = self.final_head(fusion_output)
        
        return logits

# --- Test Script ---
if __name__ == "__main__":
    batch_size = 2
    img_features = torch.randn(batch_size, 512)
    clinical_data = torch.randn(batch_size, 10)
    
    model = AdvancedClassifier(feature_dim=512, clinical_dim=10, num_classes=2)
    
    print("⏳ Testing Part 3 (Fixed): GNN-KAN Classifier...")
    output = model(img_features, clinical_data)
    
    print(f"✅ Part 3 Successful!")
    print(f"   Input Features: {img_features.shape}")
    print(f"   Output Logits: {output.shape}")