from torch.utils.data import DataLoader, Subset, random_split
import torch

def get_data_loaders(train_dev_dataset, test_dataset, batch_size=64, subset_size=None, dev_size=0.2, return_dev=False, random_seed=None):
    """
    Function to create DataLoaders for train, dev, and test datasets, or just train and test (for backward compatibility).
    
    Args:
        train_dev_dataset (Dataset): Dataset instance for both training and dev sets.
        test_dataset (Dataset): Dataset instance for testing.
        batch_size (int): Batch size for loading data.
        subset_size (int, optional): Limit the dataset size for quick testing.
        dev_size (float): Percentage of the data to use for the dev set (default is 0.2).
        return_dev (bool): Whether to return the dev loader (default is True).
        random_seed (int, optional): Random seed for reproducibility of the split.
    
    Returns:
        If return_dev is True: train_loader, dev_loader, test_loader
        If return_dev is False: train_loader, test_loader
    """
    
    # Set random seed for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Apply subset size if specified (for testing smaller dataset)
    if subset_size:
        train_dev_dataset = Subset(train_dev_dataset, range(subset_size))
        test_dataset = Subset(test_dataset, range(subset_size))
    
    # If return_dev is True, split into train and dev sets
    if return_dev:
        train_size = int((1 - dev_size) * len(train_dev_dataset))
        dev_size = len(train_dev_dataset) - train_size
        train_dataset, dev_dataset = random_split(train_dev_dataset, [train_size, dev_size])
        
        # Create DataLoader instances for train and dev sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    else:
        # No dev set, just use the entire train_dev_dataset as the training dataset
        train_loader = DataLoader(train_dev_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = None

    # Create DataLoader instance for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return loaders, handling backward compatibility
    if return_dev:
        return train_loader, dev_loader, test_loader
    else:
        return train_loader, test_loader
