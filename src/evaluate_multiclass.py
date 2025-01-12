import torch
import numpy as np
import json
from torch import nn
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from tqdm import tqdm
import os

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

    # Set up data loading and preprocessing for test images
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Define image preprocessing pipeline
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),  # Resize images to standard size
        transforms.ToTensor(),         # Convert images to PyTorch tensors
        # Normalize using ImageNet statistics for transfer learning
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Load test dataset
    test_data_dir = os.path.join(data_dir,'test')
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False,
        num_workers=max(1, os.cpu_count()-1)  # Use multiple CPU cores for data loading
    )

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

    # Calculate evaluation metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')  # Average precision across all classes
    recall = recall_score(all_labels, all_preds, average='macro')       # Average recall across all classes
    f1 = f1_score(all_labels, all_preds, average='macro')              # Average F1 across all classes
    accuracy = accuracy_score(all_labels, all_preds)                    # Overall accuracy

    # Print results to console
    print(f"\n[INFO] Multiclass Evaluation for {base_name}:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1-score (macro):  {f1:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")

    # Prepare metrics for saving
    metrics = {
        'model_name': base_name,
        'confusion_matrix': cm.tolist(),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        'accuracy': float(accuracy),
        'class_names': ['NORMAL','BACTERIA','VIRUS']
    }

    # Save metrics to JSON file
    metrics_path = os.path.join(results_dir, f"evaluation_metrics_multiclass_{base_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Multiclass metrics saved to: {metrics_path}")
    return precision, recall, f1, accuracy
