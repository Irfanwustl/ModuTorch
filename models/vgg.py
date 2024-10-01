import torch.nn as nn
from torchvision import models

def initialize_vgg16(num_classes=10, pretrained=False):
    """
    Initialize VGG16 model.
    
    Args:
    - num_classes (int): Number of output classes for the final layer (e.g., 10 for MNIST).
    - pretrained (bool): Whether to use pretrained weights (e.g., for transfer learning).

    Returns:
    - vgg16 (nn.Module): Modified VGG16 model.
    """
    # Load VGG16 with or without pre-trained weights
    if pretrained:
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)  # Use pretrained weights for transfer learning
    else:
        vgg16 = models.vgg16(weights=None)  # Initialize with random weights for pretraining

    # Modify the classifier to fit the number of output classes
    vgg16.classifier[6] = nn.Linear(4096, num_classes)  # Adjust output to `num_classes`

    return vgg16
