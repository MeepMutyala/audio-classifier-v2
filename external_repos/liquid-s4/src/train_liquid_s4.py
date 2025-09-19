import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))  # Add workspace root

import torch
import torch.nn as nn
import torch.optim as optim
from audio_utils import create_dataloaders
from liquidS4_audio import LiquidS4AudioClassifier
from tqdm import tqdm

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Liquid-S4 on {device}")
    
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        esc50_path='data/ESC-50',
        model_type='sequence', 
        batch_size=32,
        num_workers=2,
        augment=True
    )
    
    model = LiquidS4AudioClassifier(
        n_mels=128,
        num_classes=num_classes,
        d_model=64,
        n_layers=8,
        device=device
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    
    print(f"🚀 Training Liquid-S4 with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    for epoch in range(30):
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/30'):
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
            torch.save(model.state_dict(), 'liquid_s4_best.pth')
            print(f'💾 New best model saved! Val Acc: {acc:.4f}')
    
    print(f'🎯 Best Liquid-S4 accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    train()
