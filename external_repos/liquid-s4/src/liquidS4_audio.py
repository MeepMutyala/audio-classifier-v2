import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external_repos/liquid-s4/src'))

# Import directly from the src structure
from models.sequence.model import SequenceModel
from models.sequence.ss.s4 import S4
from tasks.decoders import NDDecoder

class LiquidS4AudioClassifier(nn.Module):
    """Audio classification wrapper for Liquid S4"""
    
    def __init__(self, 
                 n_mels=128, 
                 num_classes=50,
                 d_model=64,
                 n_layers=8,
                 d_state=64,
                 l_max=None,  # Will be set based on sequence length
                 dropout=0.0,
                 device=None,
                 dtype=None):
        super().__init__()
        
        # Create factory_kwargs for consistent device/dtype handling
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Input projection layer to convert mel-spectrogram features to d_model
        self.input_projection = nn.Linear(n_mels, d_model, **factory_kwargs)
        
        # Create S4 layer configuration
        s4_layer_config = {
            "_name_": "s4",
            "d_state": d_state,
            "l_max": l_max,
            "channels": 1,
            "bidirectional": False,
            "activation": "gelu",
            "postact": "glu",
            "dropout": dropout,
            "mode": "nplr",
            "measure": "legs",
            "rank": 1,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "lr": {
                "dt": 0.001,
                "A": 0.001,
                "B": 0.001
            },
            "n_ssm": 1,
            "liquid_kernel": "polyb",  # Can be set to "polyb" or "kb" for liquid variants
            "liquid_degree": 2,
            "allcombs": True,
            "lcontract": None,
            "deterministic": False,
            "verbose": True
        }
        
        # Create the S4 backbone using SequenceModel
        self.backbone = SequenceModel(
            d_model=d_model,
            n_layers=n_layers,
            transposed=True,  # (B, H, L) format
            dropout=dropout,
            tie_dropout=False,
            prenorm=True,
            n_repeat=1,
            layer=[s4_layer_config],
            residual="R",  # Residual connection
            norm="layer",  # Layer normalization
            pool=None,  # No pooling between layers
            track_norms=True,
            dropinp=0.0,
        )
        
        # Create classification decoder (NDDecoder with pooling)
        self.classifier = NDDecoder(
            d_model=d_model,
            d_output=num_classes,
            mode="pool"  # Mean pooling over sequence length
        )
        
    def forward(self, x):
        """
        Forward pass for audio classification
        Args:
            x: [batch, seq_len, n_mels] - mel-spectrogram input
        Returns:
            logits: [batch, num_classes] - classification logits
        """
        # Project input to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Pass through S4 backbone
        features, _ = self.backbone(x)  # [batch, seq_len, d_model]
        
        # Classify using NDDecoder (includes pooling)
        logits = self.classifier(features)  # [batch, num_classes]
        
        return logits

class LiquidS4AudioClassifierAdvanced(nn.Module):
    """Advanced Liquid S4 classifier with liquid kernel support"""
    
    def __init__(self, 
                 n_mels=128, 
                 num_classes=50,
                 d_model=64,
                 n_layers=8,
                 d_state=64,
                 l_max=None,
                 dropout=0.0,
                 liquid_kernel="polyb",  # "polyb" or "kb" for liquid variants
                 liquid_degree=2,
                 device=None,
                 dtype=None):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Input projection
        self.input_projection = nn.Linear(n_mels, d_model, **factory_kwargs)
        
        # Advanced S4 layer with liquid kernel
        s4_layer_config = {
            "_name_": "s4",
            "d_state": d_state,
            "l_max": l_max,
            "channels": 1,
            "bidirectional": False,
            "activation": "gelu",
            "postact": "glu",
            "dropout": dropout,
            "mode": "nplr",
            "measure": "legs",
            "rank": 1,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "lr": {
                "dt": 0.001,
                "A": 0.001,
                "B": 0.001
            },
            "n_ssm": 1,
            "liquid_kernel": liquid_kernel,  # Enable liquid kernel
            "liquid_degree": liquid_degree,
            "allcombs": True,
            "lcontract": "tanh",  # LeCun or tanh contraction
            "deterministic": False,
            "verbose": True
        }
        
        # Create backbone with liquid S4
        self.backbone = SequenceModel(
            d_model=d_model,
            n_layers=n_layers,
            transposed=True,
            dropout=dropout,
            tie_dropout=False,
            prenorm=True,
            n_repeat=1,
            layer=[s4_layer_config],
            residual="R",
            norm="layer",
            pool=None,
            track_norms=True,
            dropinp=0.0,
        )
        
        # Classification head
        self.classifier = NDDecoder(
            d_model=d_model,
            d_output=num_classes,
            mode="pool"
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        features, _ = self.backbone(x)
        logits = self.classifier(features)
        return logits
