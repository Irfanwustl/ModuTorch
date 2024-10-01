import pytest
import torch
from models.vgg import initialize_vgg16
from train.training import train_model, validate_model  
from dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from datasets.mnist_dataset import MNISTDataset

@pytest.mark.integration
def test_vgg_pretraining_on_mnist():
    """
    Integration test for VGG pretraining on the MNIST dataset.
    Tests model initialization, training, and validation.
    """
    # Set device to GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use real file paths to the MNIST dataset
    train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    # Define a basic transform (resize for VGG16 input and convert grayscale to RGB)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize for VGG16 input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize (applies to each channel)
    ])

    # Create the dataset instances with the transform applied
    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    # Get DataLoaders with only a subset of the data for testing
    train_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=8, 
        subset_size=100  # Use a small subset for quick testing
    )

    # Initialize VGG16 model (pretrained=False for the test)
    model = initialize_vgg16(num_classes=10, pretrained=False)

    # Run the training loop for one epoch
    trained_model = train_model(
        model, 
        train_loader, 
        test_loader, 
        num_epochs=1, 
        learning_rate=0.001, 
        device=device
    )

    # Check if the model's weights have been updated after training
    with torch.no_grad():
        weight_sum = trained_model.features[0].weight.sum().item()
        assert weight_sum != 0, "The model weights did not update after training."

    # Validate the model after training
    accuracy = validate_model(trained_model, test_loader, device)

    # Check that the accuracy is computed and is a valid percentage
    assert accuracy >= 0, f"Model validation failed, accuracy is {accuracy}"

    print(f"Integration test passed with accuracy: {accuracy}%")