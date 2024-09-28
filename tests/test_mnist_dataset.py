from datasets.mnist_dataset import MNISTDataset


import os
import pytest
import numpy as np
import torch
from torchvision import transforms
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
    
    # Apply a transform that includes converting to a tensor
    transform = transforms.Compose([
        transforms.ToTensor()  # Ensure output is a tensor
    ])
    
    dataset = MNISTDataset(images_filepath, labels_filepath, transform=transform)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check the shape of the image (grayscale: 1x28x28)
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor but got {type(image)}"
    assert image.shape == (1, 28, 28), f"Expected shape (1, 28, 28) but got {image.shape}"
    assert isinstance(label, int), f"Expected int but got {type(label)}"


# Test channel replication (1-channel to 3-channel conversion)
def test_rgb_conversion(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files
    
    # Apply a transform that includes converting to a tensor
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert to tensor
    ])
    
    # Create the dataset with convert_to_rgb=True
    dataset = MNISTDataset(images_filepath, labels_filepath, transform=transform, convert_to_rgb=True)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check the shape of the image (RGB: 3x28x28)
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor but got {type(image)}"
    assert image.shape == (3, 28, 28), f"Expected shape (3, 28, 28) but got {image.shape}"
    assert isinstance(label, int), f"Expected label to be an int but got {type(label)}"


# Test transformations applied correctly
def test_transforms_applied(create_temp_mnist_files):
    images_filepath, labels_filepath = create_temp_mnist_files

    # Define a simple transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to [0, 1] range
        transforms.Resize((32, 32))  # Resizes to 32x32
    ])

    dataset = MNISTDataset(images_filepath, labels_filepath, transform=transform)

    # Fetch a single image and label
    image, label = dataset[0]

    # Check the shape after resizing (should be 1x32x32 for grayscale)
    assert image.shape == (1, 32, 32)

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


