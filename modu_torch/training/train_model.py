import matplotlib.pyplot as plt  
from .plotter import Plotter  
import torch
from tqdm import tqdm

class ModelTrainer:
    """
    A class to encapsulate the training and validation functionality for a PyTorch model.
    """

    def __init__(self, model, loss_fn, optimizer, metrics, device=None):
        """
        Initialize the ModelTrainer with the given model, loss function, optimizer, and device.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            loss_fn (callable): The loss function to be used for training and validation.
            optimizer (torch.optim.Optimizer): The optimizer used for model parameter updates.
            metrics (dict): A dictionary of metrics where keys are metric names and values are callable functions.
            device (torch.device, optional): The device (e.g., 'cuda' or 'cpu') to use for computation. Defaults to 'cuda' if available.
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics  
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._train_losses = []
        self._val_losses = []
        self._metric_performances = []  
        self._predictions = None  
        self._targets = None  
        self._probabilities = None

    def train(self, train_loader, val_loader=None, num_epochs=10, **kwargs):
        """
        Train the model and optionally validate on a validation set. Optionally plot in real-time.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader, optional): DataLoader for the validation data. Defaults to None.
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
            **kwargs: Additional keyword arguments:
                - reset_losses (bool): Whether to reset losses before training. Defaults to True.
                - real_time_plot (bool): Whether to enable real-time plotting. Defaults to False.
                - plot_frequency (int): How frequently to update the plot in real-time (in epochs). Defaults to 1.
        """
        reset_losses = kwargs.get('reset_losses', True)
        real_time_plot = kwargs.get('real_time_plot', False)
        plot_frequency = kwargs.get('plot_frequency', 1)

        if reset_losses:
            self._reset_losses()

        if real_time_plot:
            plt.ion()  # Enable interactive mode for real-time plotting

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            running_metrics = {name: 0.0 for name in self.metrics}  # Initialize metrics for this epoch
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update parameters
                running_loss += loss.item()

                # Compute metrics for this batch
                for name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, targets)
                    if isinstance(metric_value, torch.Tensor):
                        running_metrics[name] += metric_value.item()
                    else:
                        running_metrics[name] += metric_value

            avg_train_loss = running_loss / len(train_loader)
            self._train_losses.append(avg_train_loss)

            # Average the metrics for the entire epoch
            avg_metrics = {name: metric_value / len(train_loader) for name, metric_value in running_metrics.items()}
            self._metric_performances.append(avg_metrics)

            # Only validate if a validation (dev) loader is provided
            if val_loader:
                avg_val_loss = self.validate(val_loader)
                self._val_losses.append(avg_val_loss[0]) ###################################

            # Real-time plotting logic
            if real_time_plot and (epoch + 1) % plot_frequency == 0:
                Plotter.plot_real_time(self._train_losses, self._val_losses, epoch + 1)

        if real_time_plot:
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Show final plot after training



    def validate(self, val_loader):
        """
        Validate the model on the validation set and capture predictions, targets, and probabilities for later use.

        Args:
            val_loader (DataLoader): DataLoader for the validation data.

        Returns:
            tuple: Validation loss averaged over all batches and a dictionary of metrics.
        """
        self.model.eval()
        val_loss = 0.0
        running_metrics = {name: 0.0 for name in self.metrics}

        all_preds = []  # To store all predictions
        all_targets = []  # To store all actual targets
        all_probs = []  # To store predicted probabilities

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()

                # Capture predictions and targets
                probs = torch.softmax(outputs, dim=1)  # Get probabilities
                _, preds = torch.max(probs, 1)  # Get the predicted classes

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())  # Store probabilities

                # Compute metrics
                for name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, targets)
                    if isinstance(metric_value, torch.Tensor):
                        running_metrics[name] += metric_value.item()
                    else:
                        running_metrics[name] += metric_value

        avg_val_loss = val_loss / len(val_loader)
        avg_metrics = {name: metric_value / len(val_loader) for name, metric_value in running_metrics.items()}

        # Store predictions, targets, and probabilities for later use
        self._predictions = all_preds
        self._targets = all_targets
        self._probabilities = all_probs

        return avg_val_loss, avg_metrics
    
    def get_probabilities(self):
        """
        Get the predicted probabilities captured during validation.

        Returns:
            list: Predicted probabilities for each class.
        """
        if self._probabilities is None:
            raise ValueError("You must run validate before getting probabilities.")
        return self._probabilities


    def get_predictions_and_targets(self):
        """
        Get the predictions and targets captured during validation.

        Returns:
            tuple: Two lists containing the predicted and actual labels.
        """
        if self._predictions is None or self._targets is None:
            raise ValueError("You must run validate before getting predictions and targets.")
        return self._predictions, self._targets
    
    def plot_loss_vs_epochs(self):
        """
        Plot the training and validation losses after training has completed.
        """
        Plotter.plot_loss_vs_epochs(self._train_losses, self._val_losses)

    def get_train_losses(self):
        """
        Getter method to retrieve the list of training losses.
        """
        return self._train_losses

    def get_val_losses(self):
        """
        Getter method to retrieve the list of validation losses.
        """
        return self._val_losses

    def get_metric_performances(self):
        """
        Getter method to retrieve the list of metric performances for each epoch.
        """
        return self._metric_performances

    def _reset_losses(self):
        """
        Reset the training and validation losses before a new training session.
        """
        self._train_losses = []
        self._val_losses = []
        self._metric_performances = []  # Reset the metrics performance list
