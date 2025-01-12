import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.data_loader import get_data_loaders
from src.model import PneumoNet

import datetime
import os
import json

# All hyperparameters in one place
HYPERPARAMETERS = {
    # Training parameters
    'epochs': 5,
    'learning_rate': 0.001,
    'patience': 5,  # early stopping
    'use_amp': True,  # Automatic Mixed Precision
    
    # Data loading parameters
    'batch_size': 32,
    'num_workers': max(1, os.cpu_count() - 1),  # Use all cores except one
    'balance_train': False,  # Use balanced sampling for training
    'augment_train': True,  # Use data augmentation
    'random_crop': False,
    'color_jitter': False,
    'desired_total_samples': None  # Will be set to dataset size if None
}

def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

def get_checkpoint_name(epoch: int, metrics: dict, timestamp: str) -> str:
    """Generate checkpoint filename with key metrics"""
    return f"checkpoint_e{epoch}_loss{metrics['val_loss']:.4f}_acc{metrics['val_accuracy']:.4f}_{timestamp}.pth"

def save_checkpoint(model, optimizer, epoch: int, metrics: dict, 
                   save_dir: str, timestamp: str, is_best: bool = False):
    """
    Spara en checkpoint av modellen. Om det är den bästa modellen hittills (is_best=True),
    sparas den också som best_model med beskrivande namn innehållande epoch och metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare model info
    model_info = {
        'epoch': epoch,
        'hyperparameters': HYPERPARAMETERS,
        'metrics': metrics,
        'timestamp': timestamp  # Lägg till timestamp i model info
    }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': timestamp  # Lägg till timestamp i checkpoint
    }
    
    # Spara checkpoint med timestamp för att gruppera per träningssession
    checkpoint_name = get_checkpoint_name(epoch, metrics, timestamp)
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Saved checkpoint to: {checkpoint_path}")
    
    # Om detta är den bästa modellen hittills, uppdatera best_model
    if is_best:
        # Skapa beskrivande filnamn med epoch och metrics
        best_name = f"best_model_e{epoch}_loss{metrics['val_loss']:.4f}_acc{metrics['val_accuracy']:.4f}"
        best_path = os.path.join(save_dir, f"{best_name}.pth")
        best_info_path = os.path.join(save_dir, f"{best_name}_info.json")
        
        # Ta bort eventuella tidigare best model filer
        for old_file in os.listdir(save_dir):
            if old_file.startswith("best_model_") and (old_file.endswith(".pth") or old_file.endswith("_info.json")):
                old_path = os.path.join(save_dir, old_file)
                try:
                    os.remove(old_path)
                    print(f"[INFO] Removed previous best model file: {old_file}")
                except Exception as e:
                    print(f"[WARNING] Could not remove old file {old_file}: {e}")
        
        # Spara nya best model filer
        torch.save(checkpoint, best_path)
        with open(best_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
            
        print(f"[INFO] Saved new best model to: {best_path}")
        print(f"[INFO] Saved model info to: {best_info_path}")

def train_model(data_dir: str, save_dir: str = None, results_dir: str = None):
    """
    Train the PneumoNet model.
    
    Args:
        data_dir (str): Directory containing the dataset
        save_dir (str, optional): Directory to save model checkpoints
        results_dir (str, optional): Directory to save training results
    """
    # Setup directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    if save_dir is None:
        save_dir = os.path.join(base_dir, "saved_models")
    if results_dir is None:
        results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Skapa beskrivande namn för training logs
    logs_path = os.path.join(results_dir, f"training_logs_e{HYPERPARAMETERS['epochs']}_b{HYPERPARAMETERS['batch_size']}_{timestamp}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if device.type == 'cuda':
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

    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement_count = 0

    print(f"[INFO] Training with hyperparameters:")
    for param, value in HYPERPARAMETERS.items():
        print(f"  {param}: {value}")
    print(f"[INFO] Logs will be saved to: {logs_path}")

    # History for plotting
    history = {
        'train_loss': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'train_accuracy': [],
        'train_confusion_matrix': [],
        'val_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_accuracy': [],
        'val_confusion_matrix': []
    }

    # Pre-fetch next batch while current batch is training
    torch.cuda.empty_cache()  # Clear any unused memory
    
    print("\n[INFO] Starting training...")
    for epoch in range(HYPERPARAMETERS['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{HYPERPARAMETERS['epochs']} ===")
        
        model.train()
        train_losses = []
        train_preds = []
        train_labels_list = []
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if HYPERPARAMETERS['use_amp']:
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
            
            # Samla metrics för training
            train_losses.append(loss.item())
            preds = (outputs > 0).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation loop med progress bar
        model.eval()
        val_losses = []
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels.float())
                val_losses.append(loss.item())
                
                preds = (outputs > 0).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Beräkna alla metrics
        train_cm = confusion_matrix(train_labels_list, train_preds).tolist()
        train_precision = precision_score(train_labels_list, train_preds)
        train_recall = recall_score(train_labels_list, train_preds)
        train_f1 = f1_score(train_labels_list, train_preds)
        train_accuracy = accuracy_score(train_labels_list, train_preds)
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        val_cm = confusion_matrix(val_labels_list, val_preds).tolist()
        val_precision = precision_score(val_labels_list, val_preds)
        val_recall = recall_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds)
        val_accuracy = accuracy_score(val_labels_list, val_preds)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Uppdatera history
        history['train_loss'].append(avg_train_loss)
        history['train_precision'].append(float(train_precision))
        history['train_recall'].append(float(train_recall))
        history['train_f1'].append(float(train_f1))
        history['train_accuracy'].append(float(train_accuracy))
        history['train_confusion_matrix'].append(train_cm)
        
        history['val_loss'].append(avg_val_loss)
        history['val_precision'].append(float(val_precision))
        history['val_recall'].append(float(val_recall))
        history['val_f1'].append(float(val_f1))
        history['val_accuracy'].append(float(val_accuracy))
        history['val_confusion_matrix'].append(val_cm)
        
        # Skriv ut metrics
        print(f"\nMetrics for epoch {epoch+1}:")
        print("Training:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1-score: {train_f1:.4f}")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  Confusion Matrix:\n{np.array(train_cm)}")
        
        print("\nValidation:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  F1-score: {val_f1:.4f}")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  Confusion Matrix:\n{np.array(val_cm)}")
        
        # Spara metrics för denna epoch
        epoch_metrics = {
            'train_loss': avg_train_loss,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_accuracy': val_accuracy,
            'train_confusion_matrix': train_cm,
            'val_confusion_matrix': val_cm
        }
        
        # Spara checkpoint och eventuellt bästa modellen
        is_best = avg_val_loss < best_val_loss
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=epoch_metrics,
            save_dir=save_dir,
            timestamp=timestamp,
            is_best=is_best
        )
        
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            no_improvement_count = 0
            print(f"\n[INFO] New best model saved! (val_loss: {best_val_loss:.4f})")
        else:
            no_improvement_count += 1
            if no_improvement_count >= HYPERPARAMETERS['patience']:
                print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs")
                break
            else:
                print(f"\n[INFO] No improvement for {no_improvement_count} epochs")

        # Spara history efter varje epoch
        with open(logs_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if epoch < HYPERPARAMETERS['epochs'] - 1:
            print("\n[INFO] Starting next epoch...")

    print("\n[INFO] Training completed!")
    return model, os.path.join(save_dir, f"best_model_{timestamp}.pth"), logs_path
