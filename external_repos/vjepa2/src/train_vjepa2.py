import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import matplotlib.pyplot as plt
import json
from datetime import datetime
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

class TrainingLogger:
    def __init__(self, model_name, save_dir="training_logs"):
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.metrics = {
            'epoch': [], 'train_loss': [], 'val_accuracy': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'learning_rate': []
        }
        
        self.log_file = os.path.join(save_dir, f"{model_name}_{self.timestamp}_metrics.json")
        self.plot_file = os.path.join(save_dir, f"{model_name}_{self.timestamp}_plots.png")
        
    def log_epoch(self, epoch, train_loss, val_acc, val_f1, val_prec, val_rec, lr):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_accuracy'].append(val_acc)
        self.metrics['val_f1'].append(val_f1)
        self.metrics['val_precision'].append(val_prec)
        self.metrics['val_recall'].append(val_rec)
        self.metrics['learning_rate'].append(lr)
        
        # Save after each epoch
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def create_plots(self):
        if not self.metrics['epoch']:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} Training Progress', fontsize=16)
        
        epochs = self.metrics['epoch']
        
        # Training Loss
        ax1.plot(epochs, self.metrics['train_loss'], 'b-', linewidth=2, marker='o')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Validation Accuracy
        ax2.plot(epochs, self.metrics['val_accuracy'], 'g-', linewidth=2, marker='s')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.set_ylim(0, 1)
        
        # All metrics comparison
        ax3.plot(epochs, self.metrics['val_accuracy'], 'g-', label='Accuracy', linewidth=2)
        ax3.plot(epochs, self.metrics['val_f1'], 'r-', label='F1', linewidth=2)
        ax3.plot(epochs, self.metrics['val_precision'], 'b-', label='Precision', linewidth=2)
        ax3.plot(epochs, self.metrics['val_recall'], 'orange', label='Recall', linewidth=2)
        ax3.set_title('Validation Metrics Comparison')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)
        ax3.set_ylim(0, 1)
        
        # Learning Rate Schedule
        ax4.plot(epochs, self.metrics['learning_rate'], 'purple', linewidth=2)
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Plots saved: {self.plot_file}")


def train():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training V-JEPA2 Audio Classifier on {device}")
    
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
    
    logger = TrainingLogger("vjepa2")

    # V-JEPA2 style training with temporal masking
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training V-JEPA2 with {sum(p.numel() for p in model.parameters()):,} parameters")
    
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
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log all metrics
        logger.log_epoch(epoch, avg_loss, acc, f1, prec, rec, current_lr)

        print(f"Epoch {epoch}: TrainLoss={avg_loss:.4f}  ValAcc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
        
        if epoch % 5 == 0:
            logger.create_plots()

        # Scheduler step
        scheduler.step()
        
        # Checkpoint & early stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'vjepa2_audio_best.pth')
            print(f"New best model saved (ValAcc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    logger.create_plots()
    best_acc_epoch = max(range(len(logger.metrics['val_accuracy'])), key=lambda i: logger.metrics['val_accuracy'][i])
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Training plots: {logger.plot_file}")
    print(f"Training data: {logger.log_file}")
    
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
    print(f'Final V-JEPA2 Audio Classification Accuracy: {test_acc:.4f}')
    return test_acc

if __name__ == '__main__':
    train()
