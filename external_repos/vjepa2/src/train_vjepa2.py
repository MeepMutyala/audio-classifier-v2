import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score
from audio_utils import create_dataloaders
from vjepa2_audio import VJEPA2AudioClassifier
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esc50_path', type=str, default='data/ESC-50')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    
    # V-JEPA2 specific model parameters
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--tubelet_size', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 8], 
                       help='Image size as (n_mels, time_frames)')
    
    return parser.parse_args()

def train():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training V-JEPA2 Audio Classifier on {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        esc50_path=args.esc50_path,
        model_type='tubelet',
        batch_size=args.batch_size,
        num_workers=args.workers,
        augment=True
    )
    
    # Create model with V-JEPA2 innovations
    model = VJEPA2AudioClassifier(
        num_classes=num_classes,
        img_size=tuple(args.img_size),
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    # V-JEPA2 style training with temporal masking
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Training V-JEPA2 with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Training with temporal masking (V-JEPA2 innovation)
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Apply temporal masking during training (V-JEPA2 principle)
            outputs = model(batch_x, apply_temporal_mask=True)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x, apply_temporal_mask=False)  # No masking during eval
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(batch_y.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        acc = (all_preds == all_labels).float().mean().item()
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch}: TrainLoss={avg_loss:.4f}  ValAcc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
        
        # Scheduler step
        scheduler.step()
        
        # Checkpoint & early stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'vjepa2_audio_best.pth')
            print(f"💾 New best model saved (ValAcc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break
    
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
    
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
