import os
import pytest
import numpy as np
import torch
from torchvision import transforms
from modu_torch.datasets.mnist_dataset import MNISTDataset
import struct


# Create mock image and label files
@pytest.fixture
def create_temp_mnist_files(tmpdir):
    # Create temporary files for images and labels
    images_filepath = os.path.join(tmpdir, 'images.idx3-ubyte')
    labels_filepath = os.path.join(tmpdir, 'labels.idx1-ubyte')

    # Create a dummy image file with a simple header and 10 images (28x28 pixels)
    with open(images_filepath, 'wb') as img_file:
        img_file.write(struct.pack('>IIII', 2051, 10, 28, 28))  # Header for images
        img_file.write(np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8).tobytes())

    # Create a dummy label file with a simple header and 10 labels
    with open(labels_filepath, 'wb') as lbl_file:
        lbl_file.write(struct.pack('>II', 2049, 10))  # Header for labels
        lbl_file.write(np.random.randint(0, 9, 10, dtype=np.uint8).tobytes())

    return images_filepath, labels_filepath


# Test dataset initialization
def test_dataset_initialization(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Create the dataset
    dataset = MNISTDataset(images_filepath, labels_filepath)

    # Test that dataset length is correct
    assert len(dataset) == 10


# Test fetching a single data point (grayscale image)
def test_getitem_single_image(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Create dataset with default transform (grayscale)
    dataset = MNISTDataset(images_filepath, labels_filepath)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check that the output is a tensor and the correct shape
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor but got {type(image)}"
    assert image.shape == (1, 28, 28), f"Expected shape (1, 28, 28) but got {image.shape}"
    assert isinstance(label, int), f"Expected int but got {type(label)}"


# Test channel replication (1-channel to 3-channel conversion)
def test_rgb_conversion(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Create the dataset with convert_to_rgb=True
    dataset = MNISTDataset(images_filepath, labels_filepath, convert_to_rgb=True)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check the shape of the image (RGB: 3x28x28)
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor but got {type(image)}"
    assert image.shape == (3, 28, 28), f"Expected shape (3, 28, 28) but got {image.shape}"
    assert isinstance(label, int), f"Expected label to be an int but got {type(label)}"


# Test transformations applied correctly
def test_transforms_applied(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Define a custom transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))  # Resize to 32x32
    ])

    dataset = MNISTDataset(images_filepath, labels_filepath, transform=transform)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check the shape after resizing (should be 1x32x32 for grayscale)
    assert image.shape == (1, 32, 32), f"Expected shape (1, 32, 32) but got {image.shape}"


# Test out-of-range index access
def test_out_of_range_index(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files
    dataset = MNISTDataset(images_filepath, labels_filepath)

    with pytest.raises(IndexError):
        _ = dataset[100]  # Accessing an index out of bounds


# Test that files are properly closed
def test_file_handling(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Create the dataset and open files
    dataset = MNISTDataset(images_filepath, labels_filepath)

    # Explicitly close the files
    dataset.close()

    # Check that the files are closed by trying to access the closed files
    with pytest.raises(ValueError):  # Expect a ValueError when accessing a closed file
        dataset.image_file.read()

    with pytest.raises(ValueError):
        dataset.label_file.read()


# Test with Transform Provided
def test_getitem_with_transform(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Define the transformation: converting to tensor and normalizing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the dataset with transform
    dataset = MNISTDataset(images_filepath, labels_filepath, transform=transform, convert_to_rgb=False)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check the shape of the tensor (grayscale: 1x28x28)
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor but got {type(image)}"
    assert image.shape == (1, 28, 28), f"Expected shape (1, 28, 28) but got {image.shape}"
    assert isinstance(label, int), f"Expected label to be an int but got {type(label)}"

    # Check if normalization was applied (since we normalized using (0.5, 0.5), mean should be close to 0)
    assert torch.allclose(image.mean(), torch.tensor(0.0), atol=1e-1), f"Normalization failed, mean is {image.mean()}"


# Test without Transform Provided (uses default transform)
def test_getitem_without_transform(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Create the dataset without providing a custom transform
    dataset = MNISTDataset(images_filepath, labels_filepath, transform=None)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check the shape of the image (grayscale: 1x28x28)
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor but got {type(image)}"
    assert image.shape == (1, 28, 28), f"Expected shape (1, 28, 28) but got {image.shape}"
    assert isinstance(label, int), f"Expected label to be an int but got {type(label)}"
