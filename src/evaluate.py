import json
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve
)
from tqdm import tqdm
import os
from src.utils import CLASS_NAMES_BINARY as CLASS_NAMES  # Updated CLASS_NAMES
from src.data_loader import get_data_loaders
from src.model import PneumoNet

# Dedicated parameters for evaluation - never use augmentation or balancing
HYPERPARAMETERS = {
    'batch_size': 128,  # Can be larger than during training since we aren't backpropagating
    'num_workers': max(1, os.cpu_count() - 1),
    'augment_train': False,  # No augmentation during evaluation
    'balance_train': False,  # No balancing either
    'desired_total_samples': None
}

def evaluate_model(model_path, data_dir, results_dir='results', plot_cm=False):
    """
    Evaluate a saved PneumoNet model on the test set.
    No plot is created, but metrics are calculated and saved in JSON.
    
    Args:
        model_path (str): Path to saved model checkpoint (pth)
        data_dir (str): Path to dataset
        results_dir (str): Directory where results (JSON) are saved
        plot_cm (bool): Ignored in this variant (no CM image is created)
    """
    # Create results_dir if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = os.path.basename(model_path).replace('.pth', '')
    print(f"[INFO] Evaluating model: {base_name}")
    print(f"[INFO] Results will be saved to: {os.path.abspath(results_dir)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and initialize model
    checkpoint = torch.load(model_path, map_location=device)
    model = PneumoNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract model info from checkpoint
    model_info = {
        'epoch': checkpoint.get('epoch', None),
        'train_loss': checkpoint.get('metrics', {}).get('train_loss', checkpoint.get('train_loss', None)),
        'val_loss': checkpoint.get('metrics', {}).get('val_loss', checkpoint.get('val_loss', None)),
        'model_name': base_name
    }

    # Get test_loader with "eval_params" (no augmentation)
    _, _, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=HYPERPARAMETERS['batch_size'],
        num_workers=HYPERPARAMETERS['num_workers'],
        desired_total_samples=HYPERPARAMETERS['desired_total_samples'],
        augment_train=HYPERPARAMETERS['augment_train'],
        balance_train=HYPERPARAMETERS['balance_train'],
        random_crop=False,
        color_jitter=False
    )

    # Verify test dataset classes match expected binary classes
    test_classes = [c for c, _ in sorted(test_loader.dataset.class_to_idx.items())]
    if test_classes != CLASS_NAMES:
        raise ValueError(f"Test dataset classes {test_classes} do not match expected binary classes {CLASS_NAMES}")

    # Inference loop
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating", leave=True)
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze()
            preds = (logits > 0).float() # threshold at 0 => sigmoid = 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            test_pbar.set_postfix({'batch_size': images.size(0)})

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_logits)

    # ROC/PR curves (only to calculate pr_auc & average_precision, but no plot)
    fpr, tpr, _ = roc_curve(all_labels, all_logits)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_logits)
    pr_auc = 0.0
    avg_precision = 0.0
    try:
        from sklearn.metrics import auc, average_precision_score
        pr_auc = auc(recall_curve, precision_curve)
        avg_precision = average_precision_score(all_labels, all_logits)
    except:
        pass

    # Print to console
    print(f"\n[INFO] Evaluation Results for {base_name}:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC-ROC:   {auc_score:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")

    # Per-class metrics
    print("\nPer-class metrics:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_precision = cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0
        class_recall = cm[i][i] / cm[:,i].sum() if cm[:,i].sum() > 0 else 0
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        print(f"{class_name}:")
        print(f"  Precision: {class_precision:.4f}")
        print(f"  Recall:    {class_recall:.4f}")
        print(f"  F1-score:  {class_f1:.4f}")

    # Save metrics to JSON
    metrics = {
        'model_name': base_name,
        'confusion_matrix': cm.tolist(),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'auc_roc': float(auc_score),
        'pr_auc': float(pr_auc),
        'average_precision': float(avg_precision),
        'model_info': model_info,
        'class_names': CLASS_NAMES,
        'per_class_metrics': {
            class_name: {
                'precision': float(cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0),
                'recall': float(cm[i][i] / cm[:,i].sum() if cm[:,i].sum() > 0 else 0),
                'f1': float(2 * (cm[i][i] / cm[i].sum()) * (cm[i][i] / cm[:,i].sum()) / 
                          ((cm[i][i] / cm[i].sum()) + (cm[i][i] / cm[:,i].sum())) 
                          if cm[i].sum() > 0 and cm[:,i].sum() > 0 else 0)
            }
            for i, class_name in enumerate(CLASS_NAMES)
        }
    }

    # Save with filename based on base_name
    metrics_path = os.path.join(results_dir, f'evaluation_metrics_{base_name}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved to: {os.path.abspath(metrics_path)}")

    print(f"\n[INFO] All evaluation results have been saved to: {os.path.abspath(results_dir)}")

    return precision, recall, f1, accuracy, auc_score, avg_precision
