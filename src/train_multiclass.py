import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix,
    precision_score,
    recall_score
)
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import datetime
import os
import json

from src.data_loader import get_data_loaders
from src.model import PneumoNet
from src.utils import print_gpu_stats, save_checkpoint

class PneumoNetMulti(nn.Module):
    """
    Multi-class variant of PneumoNet based on ResNet18 architecture.
    
    This model adapts ResNet18 for three-class classification by modifying
    the final fully connected layer to output 3 classes instead of 1000.
    
    Args:
        use_pretrained (bool): If True, initializes with ImageNet weights
    """
    def __init__(self, use_pretrained=True):
        super(PneumoNetMulti, self).__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        if use_pretrained:
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base_model = resnet18(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, 3)  # 3 classes: NORMAL/BACTERIA/VIRUS
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# Multiclass hyperparameters
MULTI_HYPERPARAMS = {
    'epochs': 30,              
    'learning_rate': 0.001,    # Initial learning rate for Adam optimizer
    'patience': 5,
    'use_amp': True,

    'batch_size': 128,        # Number of samples per batch
    'num_workers': max(1, os.cpu_count() - 1),  # DataLoader workers
    'balance_train': True,    # Whether to balance class distribution
    'augment_train': True,
    'random_crop': True, 
    'color_jitter': True,
    'desired_total_samples': None  # Target number of samples (None for all)
}

def train_epoch_multiclass(model, train_loader, val_loader, criterion, optimizer, device, epoch, scaler=None):
    """
    Executes one complete training and validation epoch for multi-class classification.

    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimization algorithm (Adam)
        device: Computing device (CPU/GPU)
        epoch: Current epoch number
        scaler: GradScaler for mixed precision training

    Returns:
        dict: Dictionary containing training and validation metrics including:
            - Losses (train/val)
            - Accuracy (train/val)
            - F1 Score (train/val)
            - Precision (train/val)
            - Recall (train/val)
            - Raw predictions and labels
    """
    model.train()
    train_losses = []
    train_preds = []
    train_labels_list = []

    train_pbar = tqdm(train_loader, desc=f"Training Multi (Epoch {epoch+1})", leave=False)
    for images, labels in train_pbar:
        images = images.to(device)
        labels = labels.to(device)  # shape [batch_size], each in [0..2]

        optimizer.zero_grad()
        if scaler is not None:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)  # shape [batch_size, 3]
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())
        preds = outputs.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels_list.extend(labels.cpu().numpy())

        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # validation
    model.eval()
    val_losses = []
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Validation Multi (Epoch {epoch+1})", leave=False)
        for images, labels in val_pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

            val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    train_acc = accuracy_score(train_labels_list, train_preds)
    val_acc   = accuracy_score(val_labels_list, val_preds)
    train_f1  = f1_score(train_labels_list, train_preds, average='macro')
    val_f1    = f1_score(val_labels_list, val_preds, average='macro')
    train_precision = precision_score(train_labels_list, train_preds, average='macro')
    val_precision = precision_score(val_labels_list, val_preds, average='macro')
    train_recall = recall_score(train_labels_list, train_preds, average='macro')
    val_recall = recall_score(val_labels_list, val_preds, average='macro')

    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'train_precision': train_precision,
        'val_precision': val_precision,
        'train_recall': train_recall,
        'val_recall': val_recall,
        # store raw labels/preds if needed
        'train_labels': train_labels_list,
        'train_preds': train_preds,
        'val_labels': val_labels_list,
        'val_preds': val_preds
    }
    return metrics

def train_model_multiclass(data_dir: str, save_dir: str=None, results_dir: str=None):
    """
    Main training function for the multi-class pneumonia classification model.

    This function handles the complete training pipeline including:
    1. Setup of directories and device configuration
    2. Data loading and preprocessing
    3. Model initialization and training loop
    4. Metrics tracking and model checkpointing
    5. Early stopping based on validation loss
    6. Results logging and saving

    Args:
        data_dir (str): Directory containing the training data
        save_dir (str, optional): Directory to save model checkpoints
        results_dir (str, optional): Directory to save training logs

    Returns:
        tuple: (
            trained model,
            path to best checkpoint,
            path to training logs
        )
    """
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.dirname(os.path.dirname(__file__))

    # default subfolder for multiclass
    if save_dir is None:
        save_dir = os.path.join(base_dir, "saved_models", "multiclass_models")
    if results_dir is None:
        results_dir = os.path.join(base_dir, "results", "multiclass")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] (Multiclass) Using device: {device}")
    if device.type == 'cuda':
        cudnn.benchmark = True
        print_gpu_stats()

    train_loader, val_loader, _ = get_data_loaders(
        data_dir=data_dir,
        batch_size=MULTI_HYPERPARAMS['batch_size'],
        num_workers=MULTI_HYPERPARAMS['num_workers'],
        augment_train=MULTI_HYPERPARAMS['augment_train'],
        random_crop=MULTI_HYPERPARAMS['random_crop'],
        color_jitter=MULTI_HYPERPARAMS['color_jitter'],
        balance_train=MULTI_HYPERPARAMS['balance_train'],
        desired_total_samples=MULTI_HYPERPARAMS['desired_total_samples']
    )

    model = PneumoNetMulti(use_pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MULTI_HYPERPARAMS['learning_rate'])
    scaler = GradScaler() if MULTI_HYPERPARAMS['use_amp'] else None

    best_val_loss = float('inf')
    best_model_path = None
    no_improvement_count = 0

    # We'll log
    temp_logs_path = os.path.join(results_dir, f"temp_training_log_multi_{start_timestamp}.json")
    history = {
        # Training metrics
        'train_loss': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'train_accuracy': [],
        'train_confusion_matrix': [],
        # Validation metrics
        'val_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_accuracy': [],
        'val_confusion_matrix': []
    }

    print("\n[INFO] Starting multiclass training...")
    final_epoch_reached = 0
    for epoch in range(MULTI_HYPERPARAMS['epochs']):
        metrics = train_epoch_multiclass(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler
        )
        current_epoch_num = epoch + 1

        # parse
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        train_acc = metrics['train_accuracy']
        val_acc = metrics['val_accuracy']
        train_f1 = metrics['train_f1']
        val_f1 = metrics['val_f1']
        train_precision = metrics['train_precision']
        val_precision = metrics['val_precision']
        train_recall = metrics['train_recall']
        val_recall = metrics['val_recall']

        # confusion matrices
        t_cm = confusion_matrix(metrics['train_labels'], metrics['train_preds'])
        v_cm = confusion_matrix(metrics['val_labels'], metrics['val_preds'])

        # store history
        history['train_loss'].append(float(train_loss))
        history['train_precision'].append(float(train_precision))
        history['train_recall'].append(float(train_recall))
        history['train_f1'].append(float(train_f1))
        history['train_accuracy'].append(float(train_acc))
        history['train_confusion_matrix'].append(t_cm.tolist())
        history['val_loss'].append(float(val_loss))
        history['val_precision'].append(float(val_precision))
        history['val_recall'].append(float(val_recall))
        history['val_f1'].append(float(val_f1))
        history['val_accuracy'].append(float(val_acc))
        history['val_confusion_matrix'].append(v_cm.tolist())

        print(f"\n=== Multiclass Epoch {current_epoch_num}/{MULTI_HYPERPARAMS['epochs']} ===")
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f}, precision={train_precision:.4f}, recall={train_recall:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}, precision={val_precision:.4f}, recall={val_recall:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        # We'll only track val_loss in "metrics" for saving
        chpt_metrics = {
            'val_loss': val_loss,
            'val_f1': val_f1
        }
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=current_epoch_num,
            metrics=chpt_metrics,
            save_dir=save_dir,
            timestamp=start_timestamp,
            is_best=is_best
        )
        if is_best:
            best_val_loss = val_loss
            best_model_path = checkpoint_path
            no_improvement_count = 0
            print("[INFO] New best multi model!")
        else:
            no_improvement_count += 1
            if no_improvement_count >= MULTI_HYPERPARAMS['patience']:
                print("[INFO] Early stopping (multiclass).")
                final_epoch_reached = current_epoch_num
                break
            else:
                print(f"[INFO] No improvement for {no_improvement_count} epochs")

        # Save logs
        with open(temp_logs_path, 'w') as f:
            json.dump(history, f, indent=2)
        final_epoch_reached = current_epoch_num

    if best_model_path is None:
        best_model_path = checkpoint_path

    final_logs_filename = f"training_logs_multi_e{final_epoch_reached:02d}_{start_timestamp}.json"
    final_logs_path = os.path.join(results_dir, final_logs_filename)
    with open(final_logs_path, 'w') as f:
        json.dump(history, f, indent=2)
    if os.path.exists(temp_logs_path):
        os.remove(temp_logs_path)

    return model, best_model_path, final_logs_path
