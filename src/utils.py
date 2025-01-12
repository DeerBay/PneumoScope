import os
import json
import torch

CLASS_NAMES = ['Normal', 'Pneumonia']

def print_gpu_stats():
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, timestamp=None, is_best=False):
    """
    Save a model checkpoint with a consistent naming scheme:
      - Per epoch:  checkpoint_e{epoch}_valloss{X}_valauc{Y}_{timestamp}.pth
      - Best model: best_model_{timestamp}.pth   (always overwritten)
    
    'history' is NOT stored in the checkpoint to reduce file size.
    """
    os.makedirs(save_dir, exist_ok=True)

    val_loss_str = f"{metrics['val_loss']:.4f}"

    # Normal checkpoint, e.g. "checkpoint_e02_valloss0.1234_20250112-181355.pth"
    checkpoint_filename = (
        f"checkpoint_e{epoch:02d}"
        f"_valloss{val_loss_str}"
        f"_{timestamp}.pth"
    )
    checkpoint_path = os.path.join(save_dir, checkpoint_filename)

    # Build a dictionary with minimal data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,  # keeps val_loss, val_auc, etc
    }

    # Save normal checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    # If best => overwrite the same file name: best_model_{timestamp}.pth
    if is_best and timestamp:
        best_model_name = f"best_model_{timestamp}.pth"
        best_path = os.path.join(save_dir, best_model_name)
        torch.save(checkpoint, best_path)
        print(f"[INFO] Best model overwritten => {best_path}")
        return best_path

    return checkpoint_path
