import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, val_losses=None):
    """
    A function to plot the learning curve after training.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list, optional): List of validation losses per epoch. Defaults to None.
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')

    # Plot validation losses if available
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

    # Add labels, title, and legend
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Force x-axis to show integer ticks only
    plt.xticks(epochs)  

    # Show plot
    plt.show()
