import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
import torch.optim as optim
from audio_utils import create_dataloaders
from vjepa2_audio import VJEPA2AudioClassifier
from tqdm import tqdm

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training V-JEPA2 Audio Classifier on {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        esc50_path='data/ESC-50',
        model_type='tubelet',
        batch_size=16,  # Can increase since we're not doing 2-stage
        num_workers=2,
        augment=True
    )
    
    # Create model with V-JEPA2 innovations
    model = VJEPA2AudioClassifier(
        num_classes=num_classes,
        img_size=(128, 8),
        num_frames=16,      # Power-of-2 temporal frames
        patch_size=8,       # Clean 8x8 patches
        tubelet_size=1,     # Audio-optimized
        embed_dim=384,
        depth=8,
        num_heads=8
    ).to(device)
    
    # V-JEPA2 style training with temporal masking
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    print(f"🚀 Training V-JEPA2 with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    best_acc = 0
    epochs = 50  # Standard for ESC-50
    
    for epoch in range(epochs):
        # Training with temporal masking (V-JEPA2 innovation)
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Apply temporal masking during training (V-JEPA2 principle)
            outputs = model(batch_x, apply_temporal_mask=True)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x, apply_temporal_mask=False)  # No masking during eval
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = correct / total
        avg_loss = train_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'vjepa2_audio_best.pth')
            print(f'💾 New best model saved! Val Acc: {acc:.4f}')
        
        scheduler.step()
    
    # Final test evaluation
    model.load_state_dict(torch.load('vjepa2_audio_best.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x, apply_temporal_mask=False)
            _, predicted = torch.max(outputs, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
    
    test_acc = test_correct / test_total
    print(f'🎯 Final V-JEPA2 Audio Classification Accuracy: {test_acc:.4f}')
    return test_acc

if __name__ == '__main__':
    train()
