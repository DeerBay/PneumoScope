import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

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
        patience=5  # Early stopping: number of epochs without improvement
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

    # Get DataLoaders (example: augmentation on, random_crop off, color jitter off)
    train_loader, val_loader, _ = get_data_loaders(
        data_dir, batch_size=batch_size, augment_train=True,
        random_crop=False, color_jitter=False, num_workers=4  # Increased from 2 to 4
    )

    # Initialize ResNet18 
    # NOTE: Here we use use_pretrained=True (matches the parameter in PneumoNet)
    model = PneumoNet(num_classes=1, use_pretrained=True).to(device)

    # BCEWithLogitsLoss + Adam
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # AMP setup
    scaler = GradScaler()

    # Early stopping
    best_val_loss = float('inf')
    no_improvement_count = 0  # Counts consecutive epochs without improvement

    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    # Pre-fetch next batch while current batch is training
    torch.cuda.empty_cache()  # Clear any unused memory
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            with autocast(device_type='cuda'):
                logits = model(images).squeeze(dim=1)  # [batch_size, 1] -> [batch_size]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    logits = model(images).squeeze(dim=1)
                    loss = criterion(logits, labels)
                val_loss += loss.item()

                # Calculate validation accuracy (binary)
                preds = (logits > 0).float()  # > 0 => 1, else 0
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds)  # 0-1 binary

        epoch_time = time.time() - start_time
 
        # Print stats
        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        print_gpu_stats()  # Print GPU stats after each epoch

        # Save to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), f"saved_models/best_model_{timestamp}.pth")
            print(f"  [!] Best model saved at epoch {epoch+1}, {timestamp}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (no improvement in {patience} epochs), {timestamp}.")
                break

    # Save final model
    torch.save(model.state_dict(), f"saved_models/final_model_{timestamp}.pth")
    print(f"Training complete {timestamp}. Final model saved as 'final_model.pth'.")

    return model, history
