import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# ensure workspace root on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../..')))

from audio_utils import create_dataloaders
from liquidS4_audio import LiquidS4AudioClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esc50_path', type=str, default='data/ESC-50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()

def train():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training Liquid-S4 on {device}")

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        esc50_path=args.esc50_path,
        model_type='sequence',
        batch_size=args.batch_size,
        num_workers=args.workers,
        augment=True
    )

    model = LiquidS4AudioClassifier(
        n_mels=128,
        num_classes=num_classes,
        d_model=256,
        n_layers=12,
        device=device
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = (all_preds == all_labels).float().mean().item()
        f1 = f1_score(all_labels, all_preds, average='macro')
        prec = precision_score(all_labels, all_preds, average='macro')
        rec = recall_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch}: TrainLoss={avg_loss:.4f}  ValAcc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

        # Scheduler step
        scheduler.step()

        # Checkpoint & early stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'liquid_s4_best.pth')
            print(f"💾 New best model saved (ValAcc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    train()
