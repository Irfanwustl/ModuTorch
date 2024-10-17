import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple CNN model for MNIST
class MNISTCNN(nn.Module):
    # def __init__(self):
    #     super(MNISTCNN, self).__init__()
    #     # First convolutional layer: input channels = 1 (grayscale image), output channels = 16, kernel size = 3
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    #     # Second convolutional layer: input channels = 16, output channels = 32, kernel size = 3
    #     self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    #     # Fully connected layer 1: input = 32 * 7 * 7, output = 128
    #     self.fc1 = nn.Linear(32 * 7 * 7, 128)
    #     # Fully connected layer 2: input = 128, output = 10 (for 10 digit classes)
    #     self.fc2 = nn.Linear(128, 10)

    # def forward(self, x):
    #     # Apply first convolution, followed by ReLU and max pooling
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool2d(x, 2)  # Reduces the image size by half (14x14)
    #     # Apply second convolution, followed by ReLU and max pooling
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool2d(x, 2)  # Reduces the image size by half again (7x7)
    #     # Flatten the image for the fully connected layers
    #     x = x.view(-1, 32 * 7 * 7)
    #     # Apply the first fully connected layer with ReLU activation
    #     x = F.relu(self.fc1(x))
    #     # Output layer (no activation function because CrossEntropyLoss applies softmax)
    #     x = self.fc2(x)
    #     return x

    def __init__(self):
        super(MNISTCNN, self).__init__()
        # A single convolutional layer with 1 input channel, 8 output channels, and 3x3 kernel
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        # A single fully connected layer that maps 8*26*26 features to 10 classes
        self.fc1 = nn.Linear(8 * 26 * 26, 10)
    
    def forward(self, x):
        # Convolutional layer with ReLU activation
        x = torch.relu(self.conv1(x))
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # Fully connected layer
        x = self.fc1(x)
        return x


