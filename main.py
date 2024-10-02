import torch
from models.vgg import initialize_vgg16
from train.training import train_model, validate_model
from dataloaders.data_loaders import get_data_loaders
from torchvision import transforms
from datasets.mnist_dataset import MNISTDataset

def main():
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
        batch_size=64
    )

    # Initialize VGG16 model (pretrained=False for the test)
    model = initialize_vgg16(num_classes=10, pretrained=False)
    model = model.to(device)

    # Train the model
    train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=10,
        learning_rate=0.001,
        device=device
    )

    # Validate the model after training
    accuracy = validate_model(model, test_loader, device)
    print(f"Final Validation Accuracy on MNIST: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
