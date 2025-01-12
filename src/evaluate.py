import os
import torch
import numpy as np
import json
from torch import nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from src.data_loader import get_data_loaders
from src.model import PneumoNet
from src.utils import plot_confusion_matrix
from tqdm import tqdm

# Dedicated parameters for evaluation - never use augmentation or balancing
EVAL_PARAMS = {
    'batch_size': 128,  # Can be larger than training since we don't need backprop
    'num_workers': max(1, os.cpu_count() - 1),
    'augment_train': False,  # Never augment during evaluation
    'random_crop': False,
    'color_jitter': False,
    'balance_train': False,  # Never balance during evaluation
    'desired_total_samples': None  # Use all samples with their natural distribution
}

def evaluate_model(model_path, data_dir, results_dir=None, plot_cm=False):
    """
    Evaluate a saved PneumoNet model on the test set.
    
    Args:
        model_path (str): Path to the saved model weights
        data_dir (str): Path to the data directory
        results_dir (str, optional): Directory to save results
        plot_cm (bool): Whether to save confusion matrix plot
    
    Returns:
        tuple: (precision, recall, f1-score, accuracy)
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get test_loader with evaluation parameters
    _, _, test_loader = get_data_loaders(
        data_dir=data_dir,
        **EVAL_PARAMS
    )

    # Initialize and load the model
    model = PneumoNet(num_classes=1, use_pretrained=False).to(device)
    
    # Ladda checkpoint och extrahera model state dict
    print(f"[INFO] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Skriv ut information om checkpointen
    print(f"[INFO] Model checkpoint from epoch: {checkpoint['epoch']}")
    print(f"[INFO] Model metrics at checkpoint:")
    for key, value in checkpoint['metrics'].items():
        if not isinstance(value, list):  # Skip confusion matrices
            print(f"  {key}: {value:.4f}")
    
    model.eval()

    # Run inference
    all_preds = []
    all_labels = []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating", leave=True)
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze()
            preds = (logits > 0).float()  # binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with batch size
            test_pbar.set_postfix({'batch_size': images.size(0)})
    
    # Compute metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"[INFO] Evaluation Results:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    # Optionally save confusion matrix
    if plot_cm:
        # Load model info if available
        info_path = model_path.replace('.pth', '_info.json')
        model_info = None
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        
        # Create filename based on model name
        base_name = os.path.basename(model_path).replace('.pth', '')
        out_file = os.path.join(results_dir, f"cm_{base_name}.png")
        
        # Plot and save confusion matrix
        plot_confusion_matrix(
            cm=cm,
            save_path=out_file,
            additional_info=model_info
        )

    return precision, recall, f1, accuracy
