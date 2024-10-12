import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to prevent plot rendering during tests
import matplotlib.pyplot as plt
from train.learning_curve import plot_learning_curve

def test_plot_learning_curve():
    """
    Test if plot_learning_curve runs without errors given valid input and verify some basic properties.
    """
    # Mock data for the test
    train_losses = [0.9, 0.8, 0.7, 0.6]
    val_losses = [1.0, 0.9, 0.8, 0.7]

    # Assert that inputs are valid
    assert len(train_losses) == 4
    assert len(val_losses) == 4

    # Call the plot function (we expect no exceptions to be raised)
    plot_learning_curve(train_losses, val_losses)

    # Check if the plot contains the correct number of lines (one for train, one for val)
    ax = plt.gca()  # Get current axis
    assert len(ax.lines) == 2  # We expect 2 lines to be plotted (train and validation)

    # Close the plot to avoid blocking the test
    plt.close()
