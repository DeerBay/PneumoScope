import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from src.data_loader import get_data_loaders
from src.model import PneumoNet

def evaluate_model(model_path, data_dir, batch_size=128, plot_cm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieve the test_loader
    _, _, test_loader = get_data_loaders(
        data_dir, 
        batch_size, 
        augment_train=False
    )

    # NOTE: Changed from 'pretrained=False' to 'use_pretrained=False'
    model = PneumoNet(num_classes=1, use_pretrained=False).to(device)

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze()
            preds = (logits > 0).float()  # binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-score:  {f1:.2f}")

    if plot_cm:
        plt.figure(figsize=(5,5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = ["Normal", "Pneumonia"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    return precision, recall, f1
