import pytest
import torch
from torch.utils.data import DataLoader
from modu_torch.dataloaders.data_loaders import get_data_loaders
from modu_torch.datasets.mnist_dataset import MNISTDataset
from torchvision import transforms

# Test DataLoader with real MNIST data using the new get_data_loaders function
@pytest.mark.parametrize("batch_size", [64, 128])  # Test with different batch sizes
def test_get_data_loaders(batch_size):
    # Define file paths for MNIST dataset
    train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    # Define basic transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the dataset instances for MNIST
    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    # Get DataLoaders
    train_loader, test_loader = get_data_loaders(
        train_dataset, 
        test_dataset, 
        batch_size=batch_size
    )

    # Assert DataLoader validity and correctness
    assert isinstance(train_loader, DataLoader), "Train loader is not a DataLoader"
    assert isinstance(test_loader, DataLoader), "Test loader is not a DataLoader"

    # Check that the batch size is correct for the train loader
    batch = next(iter(train_loader))  # Get the first batch
    images, labels = batch
    assert images.size(0) == batch_size, f"Expected batch size of {batch_size}, but got {images.size(0)}"
    assert isinstance(images, torch.Tensor), "Images are not a tensor"
    assert images.min() >= -1 and images.max() <= 1, "Images are not normalized to [-1, 1]"

    # Check that labels are the correct size and type
    assert labels.size(0) == batch_size, "The number of labels does not match the batch size"
    assert isinstance(labels, torch.Tensor), "Labels are not a tensor"


# Test DataLoader with subset size
@pytest.mark.parametrize("subset_size", [100, 500])  # Test with smaller subsets of data
def test_get_data_loaders_with_subset(subset_size):
    # Define file paths for MNIST dataset
    train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    # Define basic transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the dataset instances for MNIST
    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    # Get DataLoaders with limited subset size
    train_loader, test_loader = get_data_loaders(
        train_dataset, 
        test_dataset, 
        batch_size=8,  # Small batch size for quick testing
        subset_size=subset_size  # Apply subset size
    )

    # Check the dataset size matches the subset size
    train_dataset_size = len(train_loader.dataset)
    assert train_dataset_size == subset_size, f"Expected train dataset size of {subset_size}, but got {train_dataset_size}"

    # Fetch the first batch to ensure correctness
    batch = next(iter(train_loader))
    images, labels = batch
    assert images.size(0) == 8, "Batch size is incorrect in train loader"
    assert isinstance(images, torch.Tensor), "Images are not a tensor"
    assert isinstance(labels, torch.Tensor), "Labels are not a tensor"
