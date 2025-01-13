import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import datetime
import os
import json
import shutil  

from src.data_loader import get_data_loaders
from src.model import PneumoNet
from src.utils import save_checkpoint, print_gpu_stats
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

# All hyperparameters in one place
HYPERPARAMETERS = {
    # Training parameters
    'epochs': 30,
    'learning_rate': 0.001,
    'patience': 5,  # early stopping
    'use_amp': True,  # Automatic Mixed Precision
    
    # Data loading parameters
    'batch_size': 32,
    'num_workers': max(1, os.cpu_count() - 1),  # Use all cores except one
    'balance_train': True,  # Use balanced sampling for training
    'augment_train': True,  # Use data augmentation
    'random_crop': True,
    'color_jitter': True,
    'desired_total_samples': None,  # Will be set to dataset size if None
    
    # Early stopping configuration
    'monitor': 'val_auc',      # Metric to monitor: val_loss, val_auc, val_f1
    'monitor_mode': 'max',     # 'min' for loss, 'max' for metrics
}


def train_epoch(model, train_loader, val_loader, criterion, optimizer, device, epoch, scaler=None):
    """
    Train and validate for one epoch.
    """
    model.train()
    train_losses = []
    train_probs = []  
    train_preds = []
    train_labels_list = []
    
    # Training loop
    train_pbar = tqdm(train_loader, desc=f"Training (Epoch {epoch+1})", leave=False)
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        if scaler is not None:  # Using AMP
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels.float())
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        
        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(outputs).detach()
        
        # Check for NaN values
        if torch.isnan(probs).any():
            print("[WARNING] NaN values detected in probabilities!")
            probs = torch.nan_to_num(probs, nan=0.5)  
            
        # Collect metrics
        train_losses.append(loss.item())
        train_probs.extend(probs.cpu().numpy())
        preds = (probs > 0.5).float()
        train_preds.extend(preds.cpu().numpy())
        train_labels_list.extend(labels.cpu().numpy())
        
        # Update progress bar
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Validation loop
    model.eval()
    val_losses = []
    val_probs = []  
    val_preds = []
    val_labels_list = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Validation (Epoch {epoch+1})", leave=False)
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            
            # Check for NaN values
            if torch.isnan(probs).any():
                print("[WARNING] NaN values detected in validation probabilities!")
                probs = torch.nan_to_num(probs, nan=0.5)
            
            val_losses.append(loss.item())
            val_probs.extend(probs.cpu().numpy())
            preds = (probs > 0.5).float()
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())
            
            val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Calculate metrics
    metrics = {
        'train_loss': sum(train_losses) / len(train_losses),
        'train_precision': precision_score(train_labels_list, train_preds),
        'train_recall': recall_score(train_labels_list, train_preds),
        'train_f1': f1_score(train_labels_list, train_preds),
        'train_accuracy': accuracy_score(train_labels_list, train_preds),
        'train_auc': roc_auc_score(train_labels_list, train_probs),
        
        'val_loss': sum(val_losses) / len(val_losses),
        'val_precision': precision_score(val_labels_list, val_preds),
        'val_recall': recall_score(val_labels_list, val_preds),
        'val_f1': f1_score(val_labels_list, val_preds),
        'val_accuracy': accuracy_score(val_labels_list, val_preds),
        'val_auc': roc_auc_score(val_labels_list, val_probs),
        
        'train_labels': train_labels_list,
        'train_probs': train_probs,  
        'val_labels': val_labels_list,
        'val_probs': val_probs      
    }
    
    return metrics


def train_model(data_dir: str, save_dir: str = None, results_dir: str = None):
    """
    Train the PneumoNet model.
    
    Args:
        data_dir (str): Directory containing the dataset
        save_dir (str, optional): Directory to save model checkpoints
        results_dir (str, optional): Directory to save training results
    """
    # Start-of-training timestamp
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    if save_dir is None:
        save_dir = os.path.join(base_dir, "saved_models")
    if results_dir is None:
        results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        from torch.backends import cudnn
        cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0))
        print("cuDNN Enabled:", cudnn.enabled)
        print("cuDNN Benchmark Mode:", cudnn.benchmark)
        print_gpu_stats()

    train_loader, val_loader, _ = get_data_loaders(
        data_dir=data_dir,
        batch_size=HYPERPARAMETERS['batch_size'],
        num_workers=HYPERPARAMETERS['num_workers'],
        augment_train=HYPERPARAMETERS['augment_train'],
        random_crop=HYPERPARAMETERS['random_crop'],
        color_jitter=HYPERPARAMETERS['color_jitter'],
        balance_train=HYPERPARAMETERS['balance_train'],
        desired_total_samples=HYPERPARAMETERS['desired_total_samples']
    )

    model = PneumoNet(num_classes=1, use_pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])
    scaler = GradScaler() if HYPERPARAMETERS['use_amp'] else None

    best_metric = float('-inf') if HYPERPARAMETERS['monitor_mode'] == 'max' else float('inf')
    patience_counter = 0

    print("\n[INFO] Training with hyperparameters:")
    for param, value in HYPERPARAMETERS.items():
        print(f"  {param}: {value}")

    print(f"\n[INFO] Starting training with:")
    print(f"  Monitor metric: {HYPERPARAMETERS['monitor']}")
    print(f"  Monitor mode: {HYPERPARAMETERS['monitor_mode']}")
    print(f"  Patience: {HYPERPARAMETERS['patience']}")

    # Logs under training
    temp_logs_path = os.path.join(results_dir, f"temp_training_log_{start_timestamp}.json")
    history = {
        'train_loss': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'train_accuracy': [],
        'train_auc': [],
        'train_confusion_matrix': [],
        
        'val_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_accuracy': [],
        'val_auc': [],
        'val_confusion_matrix': []
    }

    print("\n[INFO] Starting training...")
    final_epoch_reached = 0

    for epoch in range(HYPERPARAMETERS['epochs']):
        current_epoch_num = epoch + 1
        print(f"\n=== Epoch {current_epoch_num}/{HYPERPARAMETERS['epochs']} ===")
        
        # One epoch
        metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler
        )
        
        # Update history
        history['train_loss'].append(metrics['train_loss'])
        history['train_precision'].append(float(metrics['train_precision']))
        history['train_recall'].append(float(metrics['train_recall']))
        history['train_f1'].append(float(metrics['train_f1']))
        history['train_accuracy'].append(float(metrics['train_accuracy']))
        history['train_auc'].append(float(metrics['train_auc']))
        history['train_confusion_matrix'].append(
            confusion_matrix(metrics['train_labels'], (np.array(metrics['train_probs']) > 0.5).astype(int)).tolist()
        )
        
        history['val_loss'].append(metrics['val_loss'])
        history['val_precision'].append(float(metrics['val_precision']))
        history['val_recall'].append(float(metrics['val_recall']))
        history['val_f1'].append(float(metrics['val_f1']))
        history['val_accuracy'].append(float(metrics['val_accuracy']))
        history['val_auc'].append(float(metrics['val_auc']))
        history['val_confusion_matrix'].append(
            confusion_matrix(metrics['val_labels'], (np.array(metrics['val_probs']) > 0.5).astype(int)).tolist()
        )
        
        # Get current metric value
        current_metric = metrics[HYPERPARAMETERS['monitor']]
        
        # Check if metric improved
        improved = (HYPERPARAMETERS['monitor_mode'] == 'max' and current_metric > best_metric) or \
                  (HYPERPARAMETERS['monitor_mode'] == 'min' and current_metric < best_metric)
        
        if improved:
            best_metric = current_metric
            patience_counter = 0
            
            # Save checkpoint with metrics
            metrics_to_save = {
                'train_loss': metrics['train_loss'],
                'val_loss': metrics['val_loss'],
                'val_auc': metrics['val_auc'],
                'val_f1': metrics['val_f1'],
                'best_metric': best_metric,
                'monitor': HYPERPARAMETERS['monitor'],
                'monitor_mode': HYPERPARAMETERS['monitor_mode']
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                save_dir, 
                f'checkpoint_e{epoch+1:02d}_{HYPERPARAMETERS["monitor"]}{current_metric:.4f}_{start_timestamp}.pth'
            )
            save_checkpoint(model, optimizer, epoch, checkpoint_path, metrics_to_save)
            
            # Copy to best model file
            best_model_path = os.path.join(save_dir, f'best_model_{start_timestamp}.pth')
            shutil.copy2(checkpoint_path, best_model_path)
            
            print(f"\n[INFO] {HYPERPARAMETERS['monitor']} improved to {current_metric:.4f}")
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")
            print(f"[INFO] Updated best model: {best_model_path}")
        else:
            patience_counter += 1
            print(f"\n[INFO] {HYPERPARAMETERS['monitor']} did not improve from {best_metric:.4f}")
            print(f"[INFO] Patience: {patience_counter}/{HYPERPARAMETERS['patience']}")
        
        # Print some metrics
        print(f"\nMetrics for epoch {current_epoch_num}:")
        print("Training:")
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Precision: {metrics['train_precision']:.4f}")
        print(f"  Recall: {metrics['train_recall']:.4f}")
        print(f"  F1-score: {metrics['train_f1']:.4f}")
        print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  AUC: {metrics['train_auc']:.4f}")
        print(f"  Confusion Matrix:\n{np.array(history['train_confusion_matrix'][-1])}")
        
        print("\nValidation:")
        print(f"  Loss: {metrics['val_loss']:.4f}")
        print(f"  Precision: {metrics['val_precision']:.4f}")
        print(f"  Recall: {metrics['val_recall']:.4f}")
        print(f"  F1-score: {metrics['val_f1']:.4f}")
        print(f"  Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  AUC: {metrics['val_auc']:.4f}")
        print(f"  Confusion Matrix:\n{np.array(history['val_confusion_matrix'][-1])}")

        # Early stopping check
        if patience_counter >= HYPERPARAMETERS['patience']:
            print(f"\n[INFO] Early stopping triggered after {epoch + 1} epochs")
            print(f"[INFO] Best {HYPERPARAMETERS['monitor']}: {best_metric:.4f}")
            break
        
        # Save logs every epoch
        with open(temp_logs_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        final_epoch_reached = current_epoch_num

    print("\n[INFO] Training completed!")
    
    # Fallback
    if best_model_path is None or not os.path.exists(best_model_path):
        print(f"[WARNING] best_model_path is None or doesn't exist. Using last checkpoint: {checkpoint_path}")
        best_model_path = checkpoint_path

    # Rename logs with actual epoch
    final_log_filename = f"training_logs_e{final_epoch_reached:02d}_b{HYPERPARAMETERS['batch_size']}_{start_timestamp}.json"
    final_logs_path = os.path.join(results_dir, final_log_filename)
    
    with open(final_logs_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Final training logs saved to: {final_logs_path}")
    
    if os.path.exists(temp_logs_path):
        os.remove(temp_logs_path)
    
    return model, best_model_path, final_logs_path
