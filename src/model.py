import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class PneumoNet(nn.Module):
    def __init__(self, num_classes=1, use_pretrained=True):
        super(PneumoNet, self).__init__()
        # Use the new "weights" parameter instead of "pretrained"
        if use_pretrained:
            self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.resnet18(weights=None)  # Train from scratch

        in_features = self.base_model.fc.in_features
        # Replace the last layer for binary classification (1 output)
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
