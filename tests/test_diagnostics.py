import pytest
import torch
import logging
from train.diagnostics import calculate_learning_curve
from train.train_model import ModelTrainer
from dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from datasets.mnist_dataset import MNISTDataset
from utils.project_settings import set_random_seed
from train.metrics import accuracy_metric
from train.losses import get_cross_entropy_loss
from train.optimizers import get_adam_optimizer
from train.plotter import Plotter
from models.vgg import initialize_vgg16_no_dropout

from models.mnist_cnn import MNISTCNN  # Import your custom CNN model

# Set up logging
logging.basicConfig(level=logging.INFO)

@pytest.mark.parametrize("initialize_model_fn, num_channels, img_size", [
    #(initialize_vgg16_no_dropout, 3, (224, 224)),  # VGG16 expects 3 channels, image size 224x224
    
    (MNISTCNN, 1, (28, 28))                        # MNISTCNN expects 1 channel, image size 28x28
])
def test_learning_curve_with_train_dev_split(initialize_model_fn, num_channels, img_size):
    """
    Integration test for learning curve calculation using a decoupled model initialization function.
    The `initialize_model_fn` is passed dynamically to allow testing with different models.
    The `num_channels` and `img_size` are adjusted dynamically to accommodate different models.
    """
    set_random_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    # Dynamically adjust the transformation based on the model's input requirements
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=num_channels),  # Adjust number of channels
        transforms.Resize(img_size),                             # Adjust image size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
        subset_size=1000,
        return_dev=False
    )
    dev_loader = test_loader

    total_train_samples_after_split = len(train_loader.dataset)
    dev_samples = len(dev_loader.dataset)

    logging.info(f"Number of training samples after split: {total_train_samples_after_split}")
    logging.info(f"Number of validation (dev) samples: {dev_samples}")

    train_sizes_fractions = [0.1,0.3, 0.5, 0.6, 0.7, 0.8, 1]
    actual_train_sizes = [int(fraction * total_train_samples_after_split) for fraction in train_sizes_fractions]

    def get_optimizer_fn(model):
        return get_adam_optimizer(model)

    loss_fn = get_cross_entropy_loss()
    metrics = {'accuracy': accuracy_metric}

    train_sizes, train_losses, val_losses = calculate_learning_curve(
        trainer_class=ModelTrainer,
        model_init_fn=initialize_model_fn,
        optimizer_fn=get_optimizer_fn,
        train_loader=train_loader,
        val_loader=dev_loader,
        device=device,
        num_epochs=50,
        train_sizes=train_sizes_fractions,
        loss_fn=loss_fn,
        metrics=metrics
    )

    assert len(train_losses) == len(train_sizes_fractions), "Expected training losses for each train size."
    assert len(val_losses) == len(train_sizes_fractions), "Expected validation losses for each train size."

    Plotter.plot_learning_curve(actual_train_sizes, train_losses, val_losses)

    logging.info("Decoupled learning curve test passed successfully.")
