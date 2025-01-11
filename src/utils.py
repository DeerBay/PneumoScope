# utils.py

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

def save_training_logs(history, results_dir="results", filename="training_logs.json"):
    """
    Saves the training 'history' dictionary (with losses/acc) as a JSON file in results_dir with a timestamp.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Incorporate prefix + timestamp
    logs_filename = f"{filename}_{timestamp}.json"
    logs_path = os.path.join(results_dir, logs_filename)

    with open(logs_path, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"[INFO] Training logs saved to: {logs_path}")
    return logs_path

def plot_confusion_matrix(cm, classes=["Normal", "Pneumonia"], title="Confusion Matrix",
                         save_path=None, additional_info=None):
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix to plot
        classes (list): List of class names
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
        additional_info (dict, optional): Additional information to add to title
    """
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add title with additional info if provided
    if additional_info:
        title += f"\nEpoch {additional_info['epoch']}, "
        title += f"LR {additional_info['hyperparameters']['learning_rate']}, "
        title += f"Batch {additional_info['hyperparameters']['batch_size']}\n"
        title += f"Aug: {additional_info['hyperparameters']['augment_train']}, "
        title += f"Bal: {additional_info['hyperparameters']['balance_train']}"
    plt.title(title)
    
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Confusion matrix saved to: {save_path}")
    
    plt.close()
