import torch
from models.vgg import initialize_vgg16
from train.training import train_model, validate_model
from dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from datasets.mnist_dataset import MNISTDataset
from utils.project_settings import set_random_seed



def run_mnist_vgg_experiment():
    """
    Entry point for running the VGG experiment on the MNIST dataset.
    This includes training with a dev set and final evaluation on the test set.
    """
    # Set random seed for reproducibility
    set_random_seed(42)

    # Set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use file paths to MNIST data
    train_images_filepath = 'data/MNIST/train-images-idx3-ubyte'
    train_labels_filepath = 'data/MNIST/train-labels-idx1-ubyte'
    test_images_filepath = 'data/MNIST/t10k-images-idx3-ubyte'
    test_labels_filepath = 'data/MNIST/t10k-labels-idx1-ubyte'

    # Define transforms for VGG input
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create dataset instances
    train_dataset = MNISTDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_dataset = MNISTDataset(test_images_filepath, test_labels_filepath, transform=transform)

    # Get DataLoaders, including a dev set
    train_loader, dev_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=64,
        return_dev=True
    )

    # Initialize the VGG model
    model = initialize_vgg16(num_classes=10, pretrained=False)

    # Train the model using the dev set for validation
    trained_model = train_model(
        model, 
        train_loader, 
        dev_loader,  # Use the dev set during training for validation
        num_epochs=10, 
        learning_rate=0.001, 
        device=device
    )

    # After training is complete, evaluate the model on the test set
    accuracy = validate_model(trained_model, test_loader, device)

    # Print the final accuracy on the test set
    print(f"Final test accuracy: {accuracy}%")


# Automatically run the experiment when the script is executed
if __name__ == "__main__":
    run_mnist_vgg_experiment()
