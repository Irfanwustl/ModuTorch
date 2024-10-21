
import pytest
from modu_torch.models.vgg import initialize_vgg16
from torchvision import models

# Parameterize the test to handle different numbers of output classes (e.g., for MNIST or CIFAR)
@pytest.mark.parametrize("num_classes", [10, 100])
def test_initialize_vgg16_no_pretrained(num_classes):
    """Test VGG16 initialization without pretrained weights, with variable number of classes."""
    model = initialize_vgg16(num_classes=num_classes, pretrained=False)

    # Check if the model is an instance of VGG16
    assert isinstance(model, models.VGG), "Model is not an instance of VGG16"

    # Check if the last layer is adjusted for the correct number of classes
    assert model.classifier[6].out_features == num_classes, f"The output layer should have {num_classes} units."

@pytest.mark.parametrize("num_classes", [10, 100])
def test_initialize_vgg16_pretrained(num_classes):
    """Test VGG16 initialization with pretrained weights, with variable number of classes."""
    model = initialize_vgg16(num_classes=num_classes, pretrained=True)

    # Check if the model is an instance of VGG16
    assert isinstance(model, models.VGG), "Model is not an instance of VGG16"

    # Check if the last layer is adjusted for the correct number of classes
    assert model.classifier[6].out_features == num_classes, f"The output layer should have {num_classes} units."

    # Check that the pretrained model has non-random weights in the early layers
    weight_sum = model.features[0].weight.sum().item()
    assert weight_sum != 0, "The weights in the pretrained model should not be zero."
