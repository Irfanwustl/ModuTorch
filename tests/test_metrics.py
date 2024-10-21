import torch
from modu_torch.training.metrics import accuracy_metric

def test_accuracy_metric():
    """
    Unit test for the accuracy_metric function.
    Ensures that the function correctly computes the accuracy.
    """

    # Scenario 1: All predictions are correct (100% accuracy)
    outputs = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])
    labels = torch.tensor([1, 0, 1, 0])
    expected_accuracy = 1.0  # 100% correct
    accuracy = accuracy_metric(outputs, labels)
    assert accuracy == expected_accuracy, f"Expected accuracy: {expected_accuracy}, but got: {accuracy}"

    # Scenario 2: Some predictions are incorrect (75% accuracy)
    outputs = torch.tensor([[0.1, 0.9], [0.6, 0.4], [0.2, 0.8], [0.8, 0.2]])
    labels = torch.tensor([1, 1, 1, 0])
    expected_accuracy = 0.75  # 3 out of 4 are correct
    accuracy = accuracy_metric(outputs, labels)
    assert accuracy == expected_accuracy, f"Expected accuracy: {expected_accuracy}, but got: {accuracy}"

    # Scenario 3: All predictions are incorrect (0% accuracy)
    outputs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.1, 0.9]])
    labels = torch.tensor([1, 0, 1, 0])
    expected_accuracy = 0.0  # 0% correct
    accuracy = accuracy_metric(outputs, labels)
    assert accuracy == expected_accuracy, f"Expected accuracy: {expected_accuracy}, but got: {accuracy}"

