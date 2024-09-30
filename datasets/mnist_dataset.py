import numpy as np
import torch
from torch.utils.data import Dataset
import struct
from PIL import Image
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, images_filepath, labels_filepath, transform=None, convert_to_rgb=False):
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath
        self.transform = transform
        self.convert_to_rgb = convert_to_rgb

        # Open files once during initialization and keep them open
        self.image_file = open(self.images_filepath, 'rb')
        self.label_file = open(self.labels_filepath, 'rb')

        # Read headers
        magic_num, self.num_images, self.image_height, self.image_width = struct.unpack('>IIII', self.image_file.read(16))
        magic_num_labels, self.num_labels = struct.unpack('>II', self.label_file.read(8))

        if self.num_images != self.num_labels:
            raise ValueError(f"Number of images ({self.num_images}) and labels ({self.num_labels}) do not match!")

    def close(self):
        # Explicitly close the image and label files
        if hasattr(self, 'image_file') and not self.image_file.closed:
            self.image_file.close()
        if hasattr(self, 'label_file') and not self.label_file.closed:
            self.label_file.close()

    def __del__(self):
        self.close()

        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Check if the index is out of bounds
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_images} images.")

        image = self._read_image(idx)
        label = self._read_label(idx)

        # Convert NumPy image to a PIL Image (transforms expects PIL or NumPy arrays)
        image = Image.fromarray(image)

        # Convert to 3 channels if necessary (convert to RGB before applying transforms)
        if self.convert_to_rgb:
            image = image.convert("RGB")

        # Apply transformations (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)  # Transform to tensor if needed
        else:
            # If no transform is provided, explicitly convert to tensor
            image = transforms.ToTensor()(image)

        return image, label



    def _read_image(self, idx):
        # Seek to the required image's position
        self.image_file.seek(16 + idx * 28 * 28)
        image = np.frombuffer(self.image_file.read(28 * 28), dtype=np.uint8).reshape(28, 28)
        return image

    def _read_label(self, idx):
        # Seek to the required label's position
        self.label_file.seek(8 + idx)
        label = struct.unpack('B', self.label_file.read(1))[0]
        return label

    def __del__(self):
        # Ensure that files are closed when the object is deleted
        if hasattr(self, 'image_file'):
            self.image_file.close()
        if hasattr(self, 'label_file'):
            self.label_file.close()
