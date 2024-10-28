import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import confusion_matrix
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

class Plotter:
    """
    A class responsible for plotting training and validation losses in both real-time and post-training.
    """

    @staticmethod
    def plot_loss_vs_epochs(train_losses, val_losses=None):
        """
        Plot the training and validation losses after the training has completed.

        Args:
            train_losses (list): List of training losses per epoch.
            val_losses (list, optional): List of validation losses per epoch.
        """
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))

        # Plot training loss
        plt.plot(epochs, train_losses, label='Training Loss')

        # Plot validation loss if available
        if val_losses:
            plt.plot(epochs, val_losses, label='Validation Loss')

        # Set title and labels
        plt.title('Training and Validation Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Use MaxNLocator to control the number of ticks on the x-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Ensures integer ticks and avoids overcrowding

        # Add legend and grid
        plt.legend()
        plt.grid(True)
        
        # Display the plot
        plt.show()

    @staticmethod
    def plot_real_time(train_losses, val_losses, current_epoch):
        """
        Plot training and validation losses in real-time during the training process.

        Args:
            train_losses (list): List of training losses up to the current epoch.
            val_losses (list): List of validation losses up to the current epoch.
            current_epoch (int): The current epoch being processed.
        """
        plt.clf()  # Clear the current figure for real-time updates
        plt.title(f"Training and Validation Loss (Epoch {current_epoch})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Plot training loss
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")

        # Plot validation loss if available
        if val_losses:
            plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")

        plt.legend()
        plt.pause(0.001)  # Pause to allow the plot to update in real time

    @staticmethod
    def plot_learning_curve(train_sizes_actual, train_losses, val_losses):
        """
        Plot the learning curve showing how training and validation losses change with respect to the actual number of training samples.

        Args:
            train_sizes_actual (list of int): Actual number of training samples used for each subset.
            train_losses (list of float): List of training losses for each size of the training set.
            val_losses (list of float): List of validation losses for each size of the training set.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_actual, train_losses, label='Training Loss', marker='o')
        plt.plot(train_sizes_actual, val_losses, label='Validation Loss', marker='o')

        # Use Matplotlib's AutoLocator for automatic tick spacing
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Force ticks at integer values
        
        plt.title('Learning Curve: Training and Validation Loss vs Training Set Size')
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
    
    @staticmethod
    def plot_confusion_matrix(predictions, targets, class_names=None):
        """
        Plot the confusion matrix using the provided predictions and targets.

        Args:
            predictions (list): List of predicted labels.
            targets (list): List of actual labels.
            class_names (list of str, optional): List of class names for the matrix axes. Defaults to None.
        """
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    
    
        
    @staticmethod
    def plot_roc_curves(targets, probabilities, class_names):       
        """
        Plot ROC curves for binary or multi-class classification problems.

        Args:
            targets (list): List of actual labels.
            probabilities (list): List of predicted probabilities for each class.
            class_names (list of str): List of class names.
        """
        n_classes = len(class_names)

        if n_classes == 2:  # Binary classification case
            # Only one ROC curve needed
            fpr, tpr, _ = roc_curve(targets, probabilities)  # No need to index with `[:, 1]` since it's already 1D
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (area = {roc_auc:.4f})')  # Adjust precision to 4 decimal places
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for Binary Classification')
            plt.legend(loc="lower right")
            plt.show()

        else:  # Multiclass classification case
            # Binarize the targets
            targets = label_binarize(targets, classes=range(n_classes))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(targets[:, i], np.array(probabilities)[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot all ROC curves
            plt.figure(figsize=(10, 8))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.4f})')  # Adjust precision to 4 decimal places

            plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for random classifier
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for Multiclass Classification')
            plt.legend(loc="lower right")
            plt.show()
