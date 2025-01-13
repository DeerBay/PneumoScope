import os
import json
import torch

# Class names for binary and multiclass classification
CLASS_NAMES_BINARY = ['NORMAL', 'PNEUMONIA']  # Matchar mappnamnen i filsystemet
CLASS_NAMES_MULTI = ['BACTERIA', 'NORMAL', 'VIRUS']  # Matchar alfabetisk ordning från ImageFolder

def print_gpu_stats():
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.1f}MB")
            print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1024**2:.1f}MB")
    else:
        print("\n[INFO] No GPU available. Using CPU.")

def save_checkpoint(model, optimizer, epoch, checkpoint_path, metrics=None):
    """
    Save a model checkpoint.
    
    Args:
        model (nn.Module): The PyTorch model to save
        optimizer (torch.optim.Optimizer): The optimizer used for training
        epoch (int): Current epoch number
        checkpoint_path (str): Full path where to save the checkpoint
        metrics (dict, optional): Dictionary of metrics to save with the checkpoint
    
    Returns:
        str: Path to the saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path
