import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add vjepa2 src to path and import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external_repos/vjepa2/src'))

# Import directly from the src structure
from models.vision_transformer import VisionTransformer
from models.attentive_pooler import AttentiveClassifier

class VJEPA2AudioClassifier(nn.Module):
    def __init__(self,
                 num_classes=50,
                 img_size=(128, 8), # (n_mels, time_frames)
                 num_frames=19,
                 patch_size=8,
                 tubelet_size=1,
                 embed_dim=384,
                 depth=8,
                 num_heads=8):
        super().__init__()
        
        # V-JEPA2 Backbone (adapted for audio)
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=1, # 1 channel for audio vs 3 for RGB
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            handle_nonsquare_inputs=True
        )
        
        # V-JEPA2 AttentiveClassifier (as used in original)
        self.classifier = AttentiveClassifier(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=1,
            num_classes=num_classes,
        )
        
        self.embed_dim = embed_dim
        
    def forward(self, x, apply_temporal_mask=False):
        """
        Forward pass with optional V-JEPA2 temporal masking
        x: [B, 1, T, H, W] = [batch, 1, num_frames, n_mels, time_per_frame]
        """
        if apply_temporal_mask and self.training:
            # V-JEPA2 style temporal masking (25% of frames)
            B, C, T, H, W = x.shape
            num_to_mask = T // 4  # Mask 4 out of 16 frames
            mask_idxs = torch.randperm(T, device=x.device)[:num_to_mask]
            x[:, :, mask_idxs, :, :] = 0  # Zero out masked frames
        
        features = self.backbone(x)  # [B, num_patches, embed_dim]
        logits = self.classifier(features)  # [B, num_classes]
        return logits
