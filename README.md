# Audio Classifier v2

An audio classification project that implements and compares three SOTA Structured State Space Model (SSMs) architectures on the ESC-50 environmental sound classification dataset.

## Project Overview

This project explores the effectiveness of different sequence modeling approaches for audio classification:

- **Liquid S4**: Liquid Structural State Space Models with improved dynamics
- **Mamba**: Selective State Space Models with linear-time complexity
- **V-JEPA2**: Video Joint Embedding Predictive Architecture adapted for audio

All models are trained on the ESC-50 dataset, which contains 50 classes of environmental sounds.

## Project Structure

```
audio-classifier-v2/
├── audio_utils.py                    # Core audio preprocessing and dataset utilities
├── data/                            # ESC-50 dataset
│   └── ESC-50/
│       ├── audio/                   # 2000 audio files (.wav)
│       └── meta/                    # Dataset metadata and splits
├── external_repos/                  # External model implementations
│   ├── liquid-s4/                   # Liquid S4 implementation
│   ├── mamba-code/                  # Mamba implementation  
│   └── vjepa2/                      # V-JEPA2 implementation
├── Liquid-S4_training.ipynb         # Liquid S4 training notebook
├── mamba_training.ipynb             # Mamba training notebook
├── vjepa2_training.ipynb            # V-JEPA2 training notebook
└── rand_inference.py                # inference script (not working yet)
```

## Audio Utilities (`audio_utils.py`)

### Audio Processing

The `audio_utils.py` module provides comprehensive audio preprocessing and dataset management.

- **Sample Rate**: 16kHz
- **Mel Bins**: 128
- **Duration**: 5 seconds (155 time frames)
- **Augmentation**: Time shift, noise, volume scaling, masking

### Key Components

#### `ESC50Preprocessor`
- **Purpose**: Handles audio preprocessing for all models
- **Features**:
  - Mel-spectrogram conversion (128 mel bins, 16kHz sample rate)
  - Audio augmentation (time shift, noise, volume scaling)
  - Spectrogram augmentation (time/frequency masking)
  - Center crop/pad to exactly 5 seconds (155 time frames)

#### `ESC50Dataset`
- **Purpose**: PyTorch dataset wrapper for ESC-50
- **Features**:
  - Support for different model types (`sequence` vs `tubelet`)
  - Virtual augmentation (3x data expansion)
  - Automatic train/val/test splits (folds 1-3/4/5)
  - Class mapping and label encoding

#### `convert_to_tubelets()` (for V-JEPA2)
- **Purpose**: Converts mel-spectrograms to tubelet format
- **Process**: `[155, 128] → [19, 128, 8]` (19 frames × 8 time steps each)
- **Design**: Center-cropped to preserve important audio content

## Model Implementations

### 1. Liquid S4 (`external_repos/liquid-s4/`)

**Location**: `external_repos/liquid-s4/src/liquidS4_audio.py`

**Architecture**:
- Based on Liquid Structural State Space Models
- Uses S4 layers with liquid kernel dynamics
- Input projection: `[seq_len, 128] → [seq_len, d_model]`
- Classification head with global pooling

**Training Script**: `Liquid-S4_training.ipynb`

**Key Features**:
- Configurable model dimensions and layers
- Support for different liquid kernel types (KB, PolyB)
- Optimized for long sequences

### 2. Mamba (`external_repos/mamba-code/`)

**Location**: `external_repos/mamba-code/mamba_audio.py`

**Architecture**:
- Selective State Space Models with linear-time complexity
- Based on `MixerModel` from mamba-ssm
- Input projection replaces embedding layer
- Mean pooling for classification

**Training Script**: `mamba_training.ipynb`

**Key Features**:
- Efficient selective scanning mechanism
- Hardware-aware implementation
- Linear time complexity O(L) vs O(L²) for transformers

### 3. V-JEPA2 (`external_repos/vjepa2/`)

**Location**: `external_repos/vjepa2/src/vjepa2_audio.py`

**Architecture**:
- Vision Transformer adapted for audio
- Tubelet-based processing: `[1, 19, 128, 8]`
- Attentive classification head
- Originally designed for video, adapted for audio

**Training Script**: `vjepa2_training.ipynb`

**Key Features**:
- Patch-based processing similar to vision transformers
- Attentive pooling for classification
- Handles non-square inputs

## Training Scripts

### Notebooks
- **`Liquid-S4_training.ipynb`**: Complete training pipeline for Liquid S4
- **`mamba_training.ipynb`**: Complete training pipeline for Mamba
- **`vjepa2_training.ipynb`**: Complete training pipeline for V-JEPA2

### Training Features
- **Data Augmentation**: 3x virtual expansion with random augmentations
- **Cross-Validation**: Standard ESC-50 fold splits (1-3 train, 4 val, 5 test)
- **Model Checkpointing**: Best model saving based on validation accuracy
- **Progress Tracking**: Training metrics and loss visualization

## Results

The project enables comparison of three different approaches to sequence modeling for audio classification:

- **Liquid S4**: Leverages liquid neural network dynamics for improved state space modeling
- **Mamba**: Uses selective state spaces for efficient long-range dependencies
- **V-JEPA2**: Applies vision transformer architecture to audio via tubelet processing

Each model is trained with identical preprocessing and augmentation strategies for fair comparison.

## External Dependencies

- **Liquid S4**: Based on [liquid-s4](https://github.com/raminmh/liquid-s4) repository
- **Mamba**: Based on [mamba](https://github.com/state-spaces/mamba) repository  
- **V-JEPA2**: Based on [vjepa2](https://github.com/facebookresearch/vjepa2) repository

## License

This project uses external repositories with their respective licenses. Please refer to individual repository licenses for details.
