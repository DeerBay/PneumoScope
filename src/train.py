import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# AMP
from torch.amp import autocast, GradScaler
# TQDM
from tqdm import tqdm

from src.data_loader import get_data_loaders
from src.model import PneumoNet

from sklearn.metrics import accuracy_score

import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

def train_model(
        data_dir, 
        epochs=20, 
        batch_size=128, 
        learning_rate=0.001,
        patience=5,  # Early stopping patience
        lr_patience=3,  # Learning rate scheduler patience
        lr_factor=0.1  # Learning rate reduction factor
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        # Enable cuDNN auto-tuner
        cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0))
        print("cuDNN Enabled:", cudnn.enabled)
        print("cuDNN Benchmark Mode:", cudnn.benchmark)
        print_gpu_stats()

    # Get DataLoaders
    train_loader, val_loader, _ = get_data_loaders(
        data_dir, batch_size=batch_size, augment_train=True,
        random_crop=False, color_jitter=False, num_workers=4
    )

    # Initialize model
    model = PneumoNet(num_classes=1, use_pretrained=True).to(device)

    # Initialize loss, optimizer, scheduler and scaler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=lr_patience,
        verbose=True
    )
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []

        # Training loop
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(images).squeeze()
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Get predictions
            preds = (logits > 0).float()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_loop.set_postfix({'loss': loss.item()})

        # Validation loop
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                
                with torch.cuda.amp.autocast():
                    logits = model(images).squeeze()
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                preds = (logits > 0).float()
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        val_acc = accuracy_score(all_val_labels, all_val_preds)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.2e}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            # Save the best model
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

    # Save the final model
    torch.save(model.state_dict(), 'saved_models/final_model.pth')
    
    # Load the best model state if it exists
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history
