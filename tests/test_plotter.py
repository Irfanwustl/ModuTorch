import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent actual plot rendering
import matplotlib.pyplot as plt
from plotting.plotter import Plotter

@pytest.fixture
def sample_losses():
    """Fixture to provide sample training and validation losses."""
    train_losses = [0.8, 0.6, 0.4, 0.3]
    val_losses = [0.9, 0.7, 0.5, 0.35]
    return train_losses, val_losses

def test_plot_loss_vs_epochs(sample_losses):
    """
    Test that plot_loss_vs_epochs runs without errors given valid input and verify the plot elements.
    """
    train_losses, val_losses = sample_losses

    # Call the plot function (no need to check the output, just test if it runs without error)
    Plotter.plot_loss_vs_epochs(train_losses, val_losses)

    # Check the plot contains the correct number of lines (one for train, one for val)
    ax = plt.gca()  # Get current axis
    assert len(ax.lines) == 2, "Expected 2 lines (train and validation) to be plotted."

    # Verify the labels are correct
    labels = [line.get_label() for line in ax.lines]
    assert 'Training Loss' in labels, "Missing 'Training Loss' label"
    assert 'Validation Loss' in labels, "Missing 'Validation Loss' label"

    # Close the plot to free up memory
    plt.close()

def test_plot_loss_vs_epochs_no_validation(sample_losses):
    """
    Test plot_loss_vs_epochs without validation losses.
    """
    train_losses, _ = sample_losses

    # Call the plot function without validation losses
    Plotter.plot_loss_vs_epochs(train_losses)

    # Check the plot contains only 1 line (for training)
    ax = plt.gca()
    assert len(ax.lines) == 1, "Expected 1 line (train) to be plotted."

    # Verify the label is correct
    labels = [line.get_label() for line in ax.lines]
    assert 'Training Loss' in labels, "Missing 'Training Loss' label"

    # Close the plot to free up memory
    plt.close()

def test_plot_real_time(sample_losses):
    """
    Test plot_real_time function with valid input.
    """
    train_losses, val_losses = sample_losses
    current_epoch = 4

    # Call the plot_real_time function (no need to check the output, just test if it runs without error)
    Plotter.plot_real_time(train_losses, val_losses, current_epoch)

    # Check the plot contains the correct number of lines (one for train, one for val)
    ax = plt.gca()
    assert len(ax.lines) == 2, "Expected 2 lines (train and validation) to be plotted."

    # Verify the plot title includes the current epoch
    assert f"Training and Validation Loss (Epoch {current_epoch})" in ax.get_title(), "Plot title does not contain the correct epoch"

    # Close the plot to free up memory
    plt.close()
