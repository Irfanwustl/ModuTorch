from torch.utils.data import DataLoader, Subset

def get_data_loaders(train_dataset, test_dataset, batch_size=64, subset_size=None):
    """
    Function to create DataLoaders for any dataset.
    
    Args:
        train_dataset (Dataset): Training dataset instance.
        test_dataset (Dataset): Testing dataset instance.
        batch_size (int): Batch size for loading data.
        subset_size (int, optional): Limit the dataset size for quick testing.
    
    Returns:
        DataLoader: Training and testing DataLoaders.
    """
    
    # Apply subset size if specified (for testing smaller dataset)
    if subset_size:
        train_dataset = Subset(train_dataset, range(subset_size))
        test_dataset = Subset(test_dataset, range(subset_size))

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
