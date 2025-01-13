import json
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm
import os
from src.utils import CLASS_NAMES_MULTI
from src.data_loader import get_data_loaders

def evaluate_model_multiclass(model_path, data_dir, results_dir='results/multiclass'):
    """
    Evaluate a trained multiclass pneumonia model on the test dataset.
    
    This function:
    1. Loads a trained model from a checkpoint file
    2. Processes the test dataset (chest X-ray images)
    3. Makes predictions on each image
    4. Calculates key performance metrics (precision, recall, F1, accuracy)
    5. Saves the evaluation results to a JSON file
    
    Args:
        model_path (str): Path to the saved model checkpoint (.pth file)
        data_dir (str): Base directory containing the dataset (should point to 'data_multi')
        results_dir (str): Directory where evaluation metrics will be saved
    
    Returns:
        tuple: (precision, recall, f1, accuracy) scores for the model's performance
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    base_name = os.path.basename(model_path).replace('.pth','')
    print(f"[INFO] Evaluating multiclass model: {base_name}")
    print(f"[INFO] Results => {os.path.abspath(results_dir)}")

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model architecture and load trained weights
    from src.train_multiclass import PneumoNetMulti
    model = PneumoNetMulti(use_pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode

    # Get test data loader
    _, _, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2,
        augment_train=False,
        random_crop=False,
        color_jitter=False,
        balance_train=False,
        desired_total_samples=None,
        multiclass=True  # Ensure multiclass mode
    )
    
    # Verify test dataset classes match expected multiclass classes
    test_classes = [c for c, _ in sorted(test_loader.dataset.class_to_idx.items())]
    if test_classes != CLASS_NAMES_MULTI:
        raise ValueError(f"Test dataset classes {test_classes} do not match expected multiclass classes {CLASS_NAMES_MULTI}")
    
    print("\n[INFO] Evaluating multiclass model...")

    # Lists to store model predictions and true labels
    all_preds = []
    all_labels = []

    # Run inference on test dataset
    with torch.no_grad():  # Disable gradient calculation for inference
        test_pbar = tqdm(test_loader, desc="Evaluating (multiclass)", leave=True)
        for images, labels in test_pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)  # Get predicted class (highest probability)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate both macro and micro metrics
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    accuracy = accuracy_score(all_labels, all_preds)  # Same for both macro/micro

    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2]
    )

    # Print results to console
    print(f"\n[INFO] Multiclass Evaluation for {base_name}:")
    print(f"Confusion Matrix:\n{cm}")
    print("\nMacro-averaged metrics (treats all classes equally):")
    print(f"Precision:         {precision_macro:.4f}")
    print(f"Recall:           {recall_macro:.4f}")
    print(f"F1-score:         {f1_macro:.4f}")
    print("\nMicro-averaged metrics (accounts for class imbalance):")
    print(f"Precision:         {precision_micro:.4f}")
    print(f"Recall:           {recall_micro:.4f}")
    print(f"F1-score:         {f1_micro:.4f}")
    print(f"\nAccuracy:          {accuracy:.4f}")
    
    print("\nPer-class metrics:")
    for i, class_name in enumerate(CLASS_NAMES_MULTI):
        print(f"\n{class_name}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall:    {recall_per_class[i]:.4f}")
        print(f"  F1-score:  {f1_per_class[i]:.4f}")

    # Prepare metrics for saving
    metrics_dict = {
        'accuracy': float(accuracy),
        'macro_metrics': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1': float(f1_macro)
        },
        'micro_metrics': {
            'precision': float(precision_micro),
            'recall': float(recall_micro),
            'f1': float(f1_micro)
        },
        'confusion_matrix': cm.tolist(),
        'class_metrics': {
            CLASS_NAMES_MULTI[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            } for i in range(len(CLASS_NAMES_MULTI))
        }
    }

    # Save metrics to JSON file
    metrics_path = os.path.join(results_dir, f"evaluation_metrics_multiclass_{base_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"[INFO] Multiclass metrics saved to: {metrics_path}")
    return precision_macro, recall_macro, f1_macro, accuracy
