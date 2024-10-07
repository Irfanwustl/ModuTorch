import torch
from tqdm import tqdm

class ModelTrainer:
    """
    A class to encapsulate the training, validation, and prediction functionality for a PyTorch model.
    """

    def __init__(self, model, loss_fn, optimizer, metrics, device=None):
        """
        Initialize the ModelTrainer with the given model, loss function, optimizer, metrics, and device.

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

    def train(self, train_loader, dev_loader=None, num_epochs=10):
        """
        Train the model for a specified number of epochs and optionally validate on a development set.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            dev_loader (DataLoader, optional): DataLoader for the validation data. Defaults to None.
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.

        Returns:
            None
        """
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

            # Training loop
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()  # Zero the gradients

                outputs = self.model(inputs)  # Forward pass
                loss = self.loss_fn(outputs, targets)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update parameters

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / len(train_loader)})

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

            # Validate after each epoch if a validation loader is provided
            if dev_loader:
                val_loss, val_metrics = self.validate(dev_loader)
                print(f"Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}")

    def validate(self, loader):
        """
        Validate the model on a given dataset.

        Args:
            loader (DataLoader): DataLoader for the validation or test data.

        Returns:
            float: Validation loss averaged over all batches.
            dict: Dictionary of validation metrics averaged over all batches.
        """
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        metric_values = {name: 0.0 for name in self.metrics}

        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)  # Compute validation loss
                val_loss += loss.item()

                # Compute metrics
                for name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, targets)
                    # Check if metric_value is a tensor, if so, convert it to a float using .item()
                    if isinstance(metric_value, torch.Tensor):
                        metric_values[name] += metric_value.item()
                    else:
                        metric_values[name] += metric_value

        val_loss /= len(loader)
        metric_values = {name: value / len(loader) for name, value in metric_values.items()}
        return val_loss, metric_values



