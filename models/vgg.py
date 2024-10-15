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

def initialize_vgg16_no_dropout(num_classes=10, pretrained=False):
    """
    Initialize VGG16 model without dropout layers.
    
    Args:
    - num_classes (int): Number of output classes for the final layer (e.g., 10 for MNIST).
    - pretrained (bool): Whether to use pretrained weights (e.g., for transfer learning).

    Returns:
    - vgg16 (nn.Module): Modified VGG16 model without dropout.
    """
    # Load VGG16 with or without pre-trained weights
    if pretrained:
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)  # Use pretrained weights for transfer learning
    else:
        vgg16 = models.vgg16(weights=None)  # Initialize with random weights for pretraining

    # Modify the classifier to remove dropout layers
    new_classifier = []
    for layer in vgg16.classifier:
        # Append only non-dropout layers to the new classifier
        if not isinstance(layer, nn.Dropout):
            new_classifier.append(layer)
    
    # Replace the classifier with the new one without dropout
    vgg16.classifier = nn.Sequential(*new_classifier)

    # Adjust the final layer to fit the number of output classes
    vgg16.classifier[-1] = nn.Linear(4096, num_classes)  # Adjust output to `num_classes`

    return vgg16
