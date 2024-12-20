import pytest
import torch
from modu_torch.models.mnist_cnn import MNISTCNN
from modu_torch.training.train_model import ModelTrainer
from modu_torch.dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from modu_torch.datasets.mnist_dataset import MNISTDataset
from modu_torch.utils.project_settings import set_random_seed
from modu_torch.training.metrics import accuracy_metric  
from modu_torch.training.losses import get_cross_entropy_loss  
from modu_torch.training.optimizers import get_adam_optimizer 
from modu_torch.training.plotter import Plotter  # Import the Plotter to use confusion matrix and ROC plotting
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

@pytest.mark.integration
def test_mnistcnn():
    """
    Integration test for MNISTCNN on the MNIST dataset.
    """
    # Set a fixed random seed for reproducibility across the entire project
    set_random_seed(42)

    # Set device to GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use real file paths to the MNIST dataset
    # train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    # train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    # test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    # test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    # for kaggle
    train_images_filepath = '/kaggle/input/mnist-dataset/train-images-idx3-ubyte/train-images-idx3-ubyte'
    train_labels_filepath = '/kaggle/input/mnist-dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = '/kaggle/input/mnist-dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = '/kaggle/input/mnist-dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    # Define a basic transform (resize for VGG16 input and convert grayscale to RGB)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert grayscale to RGB
        transforms.Resize((28, 28)),  # Resize for VGG16 input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize (applies to each channel)
    ])
    
    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    train_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=64,
        subset_size=10000,
        return_dev=False
    )
    dev_loader = test_loader

    # Initialize the model
    model = MNISTCNN()

    # Get the loss function and optimizer
    loss_fn = get_cross_entropy_loss()
    optimizer = get_adam_optimizer(model, learning_rate=0.001)

    # Define the metrics
    metrics = {
        'accuracy': accuracy_metric
    }

    # Initialize the ModelTrainer class
    trainer = ModelTrainer(model, loss_fn, optimizer, metrics, device)

    # Run validation before training
    val_loss_before_training = trainer.validate(dev_loader)
    train_loss_before_training = trainer.validate(train_loader)

    #################
    # Get predictions and true targets
    predictions, targets = trainer.get_predictions_and_targets()

    # Extract the unique class labels from the targets to ensure the class_names are in the correct order
    unique_classes = np.unique(targets)  # This will give [0, 1, 2, ..., 9] for MNIST, but dynamically extracted
    class_names = [str(c) for c in unique_classes]  # Convert to string for labeling the axes
    Plotter.plot_confusion_matrix(predictions, targets, class_names=class_names)

    # Get probabilities for ROC curve
    probabilities = trainer.get_probabilities()  # Call this after validation to get stored probabilities
    Plotter.plot_roc_curves(targets, probabilities, class_names=class_names)  # Plot ROC curves
    ##################

    # Train the model
    trainer.train(
        train_loader, 
        dev_loader,  # Pass dev_loader for validation
        num_epochs=10
    )

    # Run validation after training
    val_loss_after_training = trainer.validate(dev_loader)
    train_loss_after_training = trainer.validate(train_loader)

    # Log train and validation losses
    logging.info(f'train size= {len(train_loader.dataset)}, Val size= {len(dev_loader.dataset)}')
    logging.info(f'val loss before training = {val_loss_before_training}, val loss after training = {val_loss_after_training}')
    logging.info(f'train loss before training = {train_loss_before_training}, train loss after training = {train_loss_after_training}')

    # Plot training and validation losses over epochs
    trainer.plot_loss_vs_epochs()

    # Get predictions and true targets
    predictions, targets = trainer.get_predictions_and_targets()

    # Extract the unique class labels from the targets to ensure the class_names are in the correct order
    unique_classes = np.unique(targets)  # This will give [0, 1, 2, ..., 9] for MNIST, but dynamically extracted
    class_names = [str(c) for c in unique_classes]  # Convert to string for labeling the axes

    # Plot confusion matrix
    Plotter.plot_confusion_matrix(predictions, targets, class_names=class_names)

    # Get probabilities for ROC curve after training
    probabilities = trainer.get_probabilities()  # Call to retrieve stored probabilities
    Plotter.plot_roc_curves(targets, probabilities, class_names=class_names)  # Plot ROC curves after training
