import numpy as np
import torch

def calculate_learning_curve(trainer_class, model_init_fn, optimizer_fn, train_loader, val_loader, device, num_epochs=10, train_sizes=None, adjust_epochs=True, **trainer_kwargs):
    """
    Calculate the learning curve, showing how training and validation losses change as the size of the training set increases.
    
    Args:
        trainer_class (class): The ModelTrainer class to handle training and validation.
        model_init_fn (callable): A function to initialize a new model instance.
        optimizer_fn (callable): A function to initialize the optimizer, takes the model as input.
        train_loader (DataLoader): DataLoader for the full training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu').
        num_epochs (int): Number of epochs to train the model for each subset of the training data.
        train_sizes (list of float): List of fractions of the training dataset size to use for learning curve calculation (e.g., [0.1, 0.3, 0.5, 0.7, 1.0]).
        adjust_epochs (bool): Whether to adjust the number of epochs based on the size of the training set.
        trainer_kwargs (dict): Additional keyword arguments to pass to the ModelTrainer instance.
    
    Returns:
        train_sizes_actual (list): Actual number of training samples used for each fraction.
        train_losses (list): Training losses for each fraction of the training data.
        val_losses (list): Validation losses for each fraction of the training data.
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]  # Default fractions of the training set size

    train_losses = []
    val_losses = []
    train_sizes_actual = []
    
    # Get the full training set size
    full_train_size = len(train_loader.dataset)

    for train_size in train_sizes:
        print(f"Training with {train_size * 100:.0f}% of the training set...")

        # Reduce the size of the training set
        subset_size = int(train_size * full_train_size)
        subset_indices = np.random.choice(full_train_size, subset_size, replace=False)
        train_subset_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_loader.dataset, subset_indices),
            batch_size=train_loader.batch_size,
            shuffle=True
        )

        # Adjust the number of epochs based on the subset size (optional)
        adjusted_num_epochs = num_epochs
        if adjust_epochs:
            adjusted_num_epochs = int(num_epochs * (full_train_size / subset_size))

        print(f"Adjusted epochs: {adjusted_num_epochs} for {subset_size} samples.")

        # Reinitialize the model for each run
        model_copy = model_init_fn()  # Reinitialize the model
        optimizer = optimizer_fn(model_copy)

        # Initialize the ModelTrainer class for each run
        trainer = trainer_class(model_copy, trainer_kwargs['loss_fn'], optimizer, trainer_kwargs['metrics'], device)

        # Train the model with the current subset and adjusted epochs
        trainer.train(train_subset_loader, val_loader, num_epochs=adjusted_num_epochs)

        # Measure final training and validation losses after training
        final_train_loss = trainer.validate(train_subset_loader)[0]  # Validate on the same training subset
        final_val_loss = trainer.validate(val_loader)[0]  # Validate on the full validation set

        # Store the actual size and losses
        train_sizes_actual.append(subset_size)  # Actual training subset size
        train_losses.append(final_train_loss)  # Final training loss
        val_losses.append(final_val_loss)  # Final validation loss

    return train_sizes_actual, train_losses, val_losses
