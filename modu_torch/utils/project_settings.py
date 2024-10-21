import torch
import random
import numpy as np

def set_random_seed(seed_value):
    """
    Set random seed for reproducibility across numpy, random, and torch.
    
    Args:
        seed_value (int): The seed value to use for reproducibility.
    """
    # Set seed for the random module
    random.seed(seed_value)
    
    # Set seed for numpy operations
    np.random.seed(seed_value)
    
    # Set seed for PyTorch operations
    torch.manual_seed(seed_value)
    
    # If you are using CUDA, set the seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If you are using multi-GPU
        
    # Set the deterministic option to True for reproducibility (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
