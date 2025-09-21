import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))  # Add workspace root

import torch
import torch.nn as nn
import torch.optim as optim
from audio_utils import create_dataloaders
from vjepa2_audio import VJEPA2AudioClassifier
from tqdm import tqdm

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training V-JEPA2 on {device}")
    
    # V-JEPA2 uses tubelet format and smaller batches (memory intensive)
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        esc50_path='data/ESC-50',
        model_type='tubelet',  # Important: V-JEPA2 uses tubelets!
        batch_size=8,          # Smaller batch size for V-JEPA2
        num_workers=2,
        augment=True
    )
    
    # Create model
    model = VJEPA2AudioClassifier(
        num_classes=num_classes,
        img_size=(128, 8),  # (n_mels, time_frames) - adjust based on your mel-spec size
        num_frames=16,
        patch_size=8,
        tubelet_size=1,
        embed_dim=384,
        depth=8,
        num_heads=8
    ).to(device)
    
    # Training setup - V-JEPA2 typically uses smaller learning rates
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    
    print(f"🚀 Training V-JEPA2 with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    for epoch in range(30):  # V-JEPA2 may need more epochs
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/30'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability with V-JEPA2
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
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = correct / total
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'vjepa2_best.pth')
            print(f'💾 New best model saved! Val Acc: {acc:.4f}')
    
    print(f'🎯 Best V-JEPA2 accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    train()
