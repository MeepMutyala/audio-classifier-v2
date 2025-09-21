from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Hard-coded ESC-50 location
ESC50_ROOT = Path(__file__).parent / "data" / "ESC-50"

class ESC50Preprocessor:
    def __init__(self,
                 sample_rate=16000,
                 n_mels=128,
                 n_fft=1024,
                 hop_length=512,
                 max_length=16000*5,  # Exactly 5 seconds
                 augment=False):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.max_length = max_length
        self.augment = augment
        
        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Augmentation transforms
        if augment:
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)

    def load_and_preprocess(self, filepath):
        """Load audio and convert to mel-spectrogram"""
        filepath = Path(filepath)
        waveform, orig_sr = torchaudio.load(str(filepath))
        
        if orig_sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Apply waveform augmentations if enabled
        if self.augment:
            waveform = self.apply_waveform_augmentation(waveform)
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)

        EXPECTED_TIME_FRAMES = 155  # Based on 5 seconds at your settings
        mel_spec = self.center_crop_or_pad(mel_spec, EXPECTED_TIME_FRAMES)
        
        # Apply spectrogram augmentations if enabled
        if self.augment:
            mel_spec = self.apply_spectrogram_augmentation(mel_spec)
        
        # Transpose for sequence models: [seq_len, n_mels]
        mel_spec = mel_spec.squeeze(0).T
        return mel_spec
    
    def center_crop_or_pad(self, mel_spec, target_length):
        """
        Center crop or pad mel spectrogram to target length.
        Preserves beginning, middle, and end information symmetrically.
        """
        current_length = mel_spec.shape[1]  # Time dimension
        
        if current_length == target_length:
            return mel_spec
        
        elif current_length > target_length:
            # Center crop - remove equally from both sides
            excess = current_length - target_length
            start = excess // 2
            end = start + target_length
            return mel_spec[:, start:end]
        
        else:
            # Center pad - add equally to both sides  
            deficit = target_length - current_length
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            return torch.nn.functional.pad(mel_spec, (pad_left, pad_right))

    def apply_waveform_augmentation(self, waveform):
        """Apply augmentations appropriate for environmental sounds"""
        # Small time shift
        if torch.rand(1) < 0.3:
            shift = torch.randint(-int(0.05 * self.sr), int(0.05 * self.sr), (1,))
            waveform = torch.roll(waveform, shift.item(), dims=1)
        
        # Light noise addition
        if torch.rand(1) < 0.2:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Volume scaling
        if torch.rand(1) < 0.4:
            scale = torch.empty(1).uniform_(0.9, 1.1).to(waveform.device, waveform.dtype)
            waveform = waveform * scale
        
        return waveform
    
    def apply_spectrogram_augmentation(self, mel_spec):
        """Apply spectrogram-specific augmentations"""
        if torch.rand(1) < 0.3:
            mel_spec = self.time_mask(mel_spec)
        if torch.rand(1) < 0.3:
            mel_spec = self.freq_mask(mel_spec)
        return mel_spec

class ESC50Dataset(Dataset):
    def __init__(self, dataframe, esc50_path, preprocessor=None,
                 model_type='sequence', augment=False, augment_factor=3):
        self.df = dataframe.reset_index(drop=True)
        self.esc50_path = Path(esc50_path)
        self.preprocessor = preprocessor or ESC50Preprocessor()
        self.model_type = model_type  # 'sequence' or 'tubelet'
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1
        
        # Create class mapping
        self.classes = sorted(self.df['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Virtual expansion for augmentation
        if augment:
            self.virtual_length = len(self.df) * self.augment_factor
        else:
            self.virtual_length = len(self.df)
    
    def __len__(self):
        return self.virtual_length
    
    def __getitem__(self, idx):
        # Map virtual index to real index
        real_idx = idx % len(self.df)
        is_augmented = idx >= len(self.df)
        
        row = self.df.iloc[real_idx]
        audio_path = self.esc50_path / "audio" / row['filename']
        
        # Choose preprocessor based on augmentation
        if is_augmented:
            preprocessor = ESC50Preprocessor(augment=True)
        else:
            preprocessor = self.preprocessor
        
        # Process based on model type
        if self.model_type == 'tubelet':
            # For V-JEPA2: create tubelets
            mel_spec = preprocessor.load_and_preprocess(audio_path)
            data = self.convert_to_tubelets(mel_spec)
            # Add channel dimension: [16, 128, 20] → [1, 16, 128, 20]
            data = data.unsqueeze(0)
        else:
            # For Mamba and Liquid S4: regular mel-spectrograms
            data = preprocessor.load_and_preprocess(audio_path)
        
        label = self.class_to_idx[row['category']]
        return data, torch.tensor(label, dtype=torch.long)
    
    def convert_to_tubelets(self, mel_spec):
        """
        Audio-JEPA standard: Power-of-2 dimensions with center cropping
        mel_spec: [155, 128] → [16, 128, 8]
        """
        time_steps, n_mels = mel_spec.shape  # [155, 128]
        
        # Power-of-2 parameters (Audio-JEPA best practice)
        num_frames = 16        # 2^4 temporal frames  
        time_per_frame = 8     # 2^3 time steps per frame
        
        # Center crop to preserve important audio content
        target_length = num_frames * time_per_frame  # 128
        start_idx = (time_steps - target_length) // 2  # 13
        end_idx = start_idx + target_length            # 141
        
        mel_spec = mel_spec[start_idx:end_idx]  # [128, 128]
        
        # Reshape: [128, 128] → [16, 8, 128] → [16, 128, 8]
        frames = mel_spec.view(num_frames, time_per_frame, n_mels)
        tubelets = frames.permute(0, 2, 1)
        return tubelets


def create_esc50_splits(path):
    """Load ESC-50 metadata and split into train/val/test."""
    meta_csv = Path(path) / "meta" / "esc50.csv"
    df = pd.read_csv(meta_csv)
    train_df = df[df.fold.isin([1,2,3])]
    val_df = df[df.fold == 4]
    test_df = df[df.fold == 5]
    return train_df, val_df, test_df

def create_dataloaders(esc50_path=None, model_type='sequence', batch_size=32,
                      num_workers=4, augment=True, augment_factor=3):
    """Create DataLoaders using ESC 50 path."""

    if esc50_path is None:
        esc50_path = ESC50_ROOT

    train_df, val_df, test_df = create_esc50_splits(esc50_path)
    
    # Create preprocessors
    train_preprocessor = ESC50Preprocessor(augment=augment)
    val_preprocessor = ESC50Preprocessor(augment=False)  # No augmentation for val/test
    
    # Create datasets
    train_dataset = ESC50Dataset(
        train_df, esc50_path, train_preprocessor,
        model_type=model_type, augment=augment, augment_factor=augment_factor
    )
    
    val_dataset = ESC50Dataset(
        val_df, esc50_path, val_preprocessor,
        model_type=model_type, augment=False
    )
    
    test_dataset = ESC50Dataset(
        test_df, esc50_path, val_preprocessor,
        model_type=model_type, augment=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes
