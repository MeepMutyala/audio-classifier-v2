import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))  # Add workspace root

import torch
import torch.nn as nn
import torch.optim as optim
from audio_utils import create_dataloaders
from mamba_audio import MambaAudioClassifier  # This import will work after you copy the file
from tqdm import tqdm

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Mamba on {device}")
    
    # Load data - path is relative to workspace root
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        esc50_path='data/ESC-50',  # Relative to workspace root
        model_type='sequence',
        batch_size=16,
        num_workers=2,
        augment=True
    )
    
    # Create model
    model = MambaAudioClassifier(
        n_mels=128,
        num_classes=num_classes,
        d_model=256,  # Smaller for faster training
        n_layer=6,
        device=device
    ).to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    print(f"🚀 Training Mamba with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    for epoch in range(20):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/20'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
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
            torch.save(model.state_dict(), 'mamba_best.pth')
            print(f'💾 New best model saved! Val Acc: {acc:.4f}')
    
    print(f'🎯 Best Mamba accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    train()
