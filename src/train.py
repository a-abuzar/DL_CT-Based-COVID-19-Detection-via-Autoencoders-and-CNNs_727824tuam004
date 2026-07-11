import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import HybridCovidModel
from preprocess import get_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

try:
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
except ImportError:
    HAS_TPU = False

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=1e-4, freeze_resnet=False):
    if HAS_TPU:
        device = xm.xla_device()
        print(f"Using device: TPU ({device})")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    # Load dataloaders
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)

    # Initialize model
    model = HybridCovidModel(freeze_resnet=freeze_resnet).to(device)

    # Loss and Optimizer
    # We have two outputs: classification (BCEWithLogitsLoss) and reconstruction (MSELoss)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_recon = nn.MSELoss()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    best_val_loss = float('inf')
    metrics_history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}

    os.makedirs('../models', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            
            # Forward pass
            out, decoded = model(images)
            
            # Compute loss
            # Classification loss
            loss_cls = criterion_cls(out, labels)
            # Reconstruction loss (autoencoder)
            loss_recon = criterion_recon(decoded, images)
            
            # Combined loss
            loss = loss_cls + 0.5 * loss_recon
            
            loss.backward()
            
            if HAS_TPU:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                labels = labels.unsqueeze(1).float().to(device)

                out, decoded = model(images)
                
                loss_cls = criterion_cls(out, labels)
                loss_recon = criterion_recon(decoded, images)
                loss = loss_cls + 0.5 * loss_recon
                
                val_loss += loss.item() * images.size(0)
                
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # Ensure AUC can be calculated
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} - Precision: {prec:.4f} - Recall: {rec:.4f} - F1: {f1:.4f} - AUC: {auc:.4f}")

        # Log history
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_acc'].append(acc)
        metrics_history['val_f1'].append(f1)
        metrics_history['val_auc'].append(auc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../models/best_model.pth')
            print("--> Saved Best Model")

    # Save metrics history
    with open('../models/metrics_history.json', 'w') as f:
        json.dump(metrics_history, f)
    print("Training Complete. Metrics saved.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    
    # Check if data exists
    import glob
    if not glob.glob(os.path.join(data_dir, 'COVID', '*.*')):
        print("Dataset not found. Please place dataset in 'data/' directory.")
        print("To run a test, generate mock data using 'python create_mock_data.py'")
    else:
        # Reduced epochs for quick testing. Real training should be ~20-50 epochs.
        train_model(data_dir, num_epochs=5, batch_size=16)
