import pytest
import torch
from models.mnist_cnn import MNISTCNN
from train.train_model import ModelTrainer  # Updated to match the new import
from dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from datasets.mnist_dataset import MNISTDataset
from utils.project_settings import set_random_seed
from train.metrics import accuracy_metric  
from train.losses import get_cross_entropy_loss  
from train.optimizers import get_adam_optimizer 
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)

@pytest.mark.integration
def test_mnistcnn():
    """
    Integration test for MNISTCCNN on the MNIST dataset.
    """
    # Set a fixed random seed for reproducibility across the entire project
    set_random_seed(42)

    # Set device to GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use real file paths to the MNIST dataset
    train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    # Define a basic transform (resize for VGG16 input and convert grayscale to RGB)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert grayscale to RGB
        transforms.Resize((28, 28)),  # Resize for VGG16 input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize (applies to each channel)
    ])
    
    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    # train_loader, dev_loader, test_loader = get_data_loaders(
    #     train_dataset,
    #     test_dataset,
    #     batch_size=64,
    #     subset_size=1000,
    #     return_dev=True
    # )

    train_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=64,
        subset_size=10000,
        return_dev=False
    )
    dev_loader = test_loader

    # Initialize VGG16 model (pretrained=False for the test)
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

    val_loss_before_training = trainer.validate(dev_loader)
    train_loss_before_training = trainer.validate(train_loader)
    # Run the training loop for one epoch with the option of real-time plotting and custom plot frequency
    trainer.train(
        train_loader, 
        dev_loader,  # Pass dev_loader for validation if use_dev, else pass None
        num_epochs=10
    )

    val_loss_after_training = trainer.validate(dev_loader)
    train_loss_after_training = trainer.validate(train_loader)

    logging.info(f'train size= {len(train_loader.dataset)}, Val size= {len(dev_loader.dataset)}')
    logging.info(f'val loss before training = {val_loss_before_training}, val loss after training = {val_loss_after_training}')
    logging.info(f'train loss before training = {train_loss_before_training}, train loss after training = {train_loss_after_training}')

    trainer.plot_loss_vs_epochs()