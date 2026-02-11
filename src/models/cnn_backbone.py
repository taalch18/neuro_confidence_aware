import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def get_model(pretrained=True):
    """
    Returns a ResNet18 model with the final layer replaced for 3 classes.
    """
    # Use standard weights if pretrained=True
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Replace the fully connected layer
    # ResNet18 fc in_features is 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model
