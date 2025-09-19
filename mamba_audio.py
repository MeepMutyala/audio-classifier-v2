import torch
import torch.nn as nn
import sys
import os

# Add submodules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external_repos/mamba'))

from mamba_ssm.models.mixer_seq_simple import MixerModel

class MambaAudioClassifier(nn.Module):
    def __init__(self, 
                 n_mels=128, 
                 num_classes=50, 
                 d_model=512,
                 n_layer=12,
                 pool_method='mean',
                 device=None,
                 dtype=None):
        super().__init__()
        
        # Create factory_kwargs for consistent device/dtype handling
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Create backbone directly
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_model * 4,  # Default intermediate size
            vocab_size=50000,  # Still needed for initialization
            ssm_cfg={},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            device=device,
            dtype=dtype,
        )
        
        # REPLACE the embedding layer with linear projection
        self.backbone.embedding = nn.Linear(n_mels, d_model, **factory_kwargs)
        
        # Add classification head
        self.classification_head = nn.Linear(d_model, num_classes, **factory_kwargs)
        self.pool_method = pool_method
        
    def forward(self, x):
        # x: [batch, seq_len, n_mels]
        hidden_states = self.backbone(x)  # [batch, seq_len, d_model]
        
        # Pool over sequence dimension
        if self.pool_method == 'mean':
            pooled = hidden_states.mean(dim=1)  # [batch, d_model]
        elif self.pool_method == 'last':
            pooled = hidden_states[:, -1, :]  # [batch, d_model]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
        
        # Classify
        logits = self.classification_head(pooled)  # [batch, num_classes]
        
        return logits
