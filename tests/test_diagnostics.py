import pytest
import torch
import logging  # Add logging
from models.vgg import initialize_vgg16
from train.diagnostics import calculate_learning_curve
from train.train_model import ModelTrainer
from dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from datasets.mnist_dataset import MNISTDataset
from utils.project_settings import set_random_seed
from train.metrics import accuracy_metric
from train.losses import get_cross_entropy_loss
from train.optimizers import get_adam_optimizer
from plotting.plotter import Plotter

# Set up logging
logging.basicConfig(level=logging.INFO)  # This will allow INFO level messages to be printed

@pytest.mark.integration
def test_vgg_learning_curve_with_train_dev_split():
    """
    Integration test for learning curve calculation using a decoupled model initialization function.
    Tests model initialization, learning curve computation, and plotting for different training set sizes,
    accounting for the train-dev split.
    """
    set_random_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    train_loader, dev_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=8,
        subset_size=100,
        return_dev=True
    )

    total_train_samples_after_split = len(train_loader.dataset)
    dev_samples = len(dev_loader.dataset)

    # Use logging instead of print
    logging.info(f"Number of training samples after split: {total_train_samples_after_split}")
    logging.info(f"Number of validation (dev) samples: {dev_samples}")

    train_sizes_fractions = [0.1, 0.3, 0.5]

    actual_train_sizes = [int(fraction * total_train_samples_after_split) for fraction in train_sizes_fractions]
    logging.info(f"Actual number of training samples used at each point: {actual_train_sizes}")

    def initialize_vgg_model():
        return initialize_vgg16(num_classes=10, pretrained=False)

    def get_optimizer_fn(model):
        return get_adam_optimizer(model)

    loss_fn = get_cross_entropy_loss()
    metrics = {'accuracy': accuracy_metric}

    train_sizes, train_losses, val_losses = calculate_learning_curve(
        trainer_class=ModelTrainer,
        model_init_fn=initialize_vgg_model,
        optimizer_fn=get_optimizer_fn,
        train_loader=train_loader,
        val_loader=dev_loader,
        device=device,
        num_epochs=1,
        train_sizes=train_sizes_fractions,
        loss_fn=loss_fn,
        metrics=metrics
    )

    assert len(train_losses) == len(train_sizes_fractions), "Expected training losses for each train size."
    assert len(val_losses) == len(train_sizes_fractions), "Expected validation losses for each train size."

    Plotter.plot_learning_curve(actual_train_sizes, train_losses, val_losses)

    logging.info("Decoupled learning curve test passed successfully.")
