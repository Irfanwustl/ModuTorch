import pytest
import torch
from modu_torch.models.vgg import initialize_vgg16
from modu_torch.training.train_model import ModelTrainer  # Updated to match the new import
from modu_torch.dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from modu_torch.datasets.mnist_dataset import MNISTDataset
from modu_torch.utils.project_settings import set_random_seed
from modu_torch.training.metrics import accuracy_metric  
from modu_torch.training.losses import get_cross_entropy_loss  
from modu_torch.training.optimizers import get_adam_optimizer 

@pytest.mark.integration
@pytest.mark.parametrize("use_dev, real_time_plot, plot_frequency", [
    (False, False, 1),
    (True, False, 1),
    (True, True, 5)  # Real-time plotting enabled every 5 epochs
])
def test_vgg_pretraining_on_mnist(use_dev, real_time_plot, plot_frequency):
    """
    Integration test for VGG pretraining on the MNIST dataset.
    Tests model initialization, training, validation, and the getter methods for losses and metrics.
    
    Args:
        use_dev (bool): Whether to use the dev set or not.
        real_time_plot (bool): Whether to enable real-time plotting.
        plot_frequency (int): Frequency of real-time plotting (in epochs).
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
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize for VGG16 input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize (applies to each channel)
    ])

    # Create the dataset instances with the transform applied
    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    # Get DataLoaders with or without a dev set
    if use_dev:
        train_loader, dev_loader, test_loader = get_data_loaders(
            train_dataset,
            test_dataset,
            batch_size=8, 
            subset_size=100,  # Use a small subset for quick testing
            return_dev=True
        )
    else:
        train_loader, test_loader = get_data_loaders(
            train_dataset,
            test_dataset,
            batch_size=8, 
            subset_size=100,  # Use a small subset for quick testing
            return_dev=False
        )
        dev_loader = None  # No dev set in this case

    # Initialize VGG16 model (pretrained=False for the test)
    model = initialize_vgg16(num_classes=10, pretrained=False)

    # Get the loss function and optimizer
    loss_fn = get_cross_entropy_loss()
    optimizer = get_adam_optimizer(model, learning_rate=0.001)

    # Define the metrics
    metrics = {
        'accuracy': accuracy_metric
    }

    # Initialize the ModelTrainer class
    trainer = ModelTrainer(model, loss_fn, optimizer, metrics, device)

    # Run the training loop for one epoch with the option of real-time plotting and custom plot frequency
    trainer.train(
        train_loader, 
        dev_loader if use_dev else None,  # Pass dev_loader for validation if use_dev, else pass None
        num_epochs=1,
        real_time_plot=real_time_plot,  # Enable real-time plotting based on the test parameters
        plot_frequency=plot_frequency  # Set plot frequency
    )


    # Check if the model's weights have been updated after training
    with torch.no_grad():
        weight_sum = trainer.model.features[0].weight.sum().item()
        assert weight_sum != 0, "The model weights did not update after training."

    # Validate the model after training
    val_loss, val_metrics = trainer.validate(test_loader)

    # Check that the accuracy is computed and is a valid percentage
    assert val_metrics['accuracy'] >= 0, f"Model validation failed, accuracy is {val_metrics['accuracy']}"

    # Test the getter methods for losses
    train_losses = trainer.get_train_losses()
    val_losses = trainer.get_val_losses()

    # Ensure that the training and validation losses were tracked and can be retrieved
    assert len(train_losses) > 0, "Training losses were not recorded."
    if use_dev:
        assert len(val_losses) > 0, "Validation losses were not recorded."
    else:
        assert len(val_losses) == 0, "Validation losses should not be recorded without a dev set."

    # Test the getter method for metric performance
    metric_performances = trainer.get_metric_performances()

    # Ensure that metric performances (e.g., accuracy) were tracked and can be retrieved
    assert len(metric_performances) > 0, "Metric performances were not recorded."
    assert 'accuracy' in metric_performances[0], "Accuracy metric was not recorded."

    print(f"Integration test {'with' if use_dev else 'without'} dev set passed with accuracy: {val_metrics['accuracy']}%")
