# ModuTorch

ModuTorch is a modular deep learning framework built using PyTorch. Designed with flexibility in mind, the project allows easy integration of various datasets and models for different machine learning tasks. The framework is guided by best practices from Andrew Ng's *Machine Learning Yearning*.

## Overview
ModuTorch provides a foundation for building, training, and evaluating deep learning models in PyTorch. The framework is adaptable, enabling quick experimentation with different datasets and architectures. **MNIST** is currently used for testing purposes, but the project is designed to easily support other datasets as well.

## Current Features:
- **Dataset Flexibility**: Tested with the MNIST dataset, but designed to integrate other datasets seamlessly.
- **Modular Design**: The code is modular, allowing for easy extension to other datasets and models.
- **Training and Validation**: Supports training on one train set and one dev set.
- **Evaluation Metrics**: Includes plotting tools for confusion matrices and ROC curves for performance evaluation.

## To-Do (following Andrew Ng’s *Machine Learning Yearning*):
1. **Error Analysis**: Implement error analysis to identify common error patterns and improve model performance.
2. **Improved Data Splitting**: Currently, the dataset is split into one train set and one dev set. The goal is to implement a more sophisticated strategy with **two dev sets** for more robust evaluation.
3. **Advanced Hyperparameter Tuning**: Introduce more advanced hyperparameter tuning techniques to optimize model performance.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Irfanwustl/ModuTorch.git
    cd ModuTorch
    ```

2. **Install dependencies**:

    ```bash
    TODO
    #pip install -r requirements.txt
    ```

3. **Run the project**:
    ```bash
    TODO
    ```

## Usage
### Training
You can train a model on the MNIST dataset using the default configuration. The framework is designed for flexibility, so you can easily modify it to work with other datasets.

### Evaluation
The model's performance is evaluated on the dev set, with results visualized using:
- Confusion Matrix
- ROC Curves (for multiclass classification)

## Future Plans:
- **Error Analysis**: Implement tools for error analysis to better understand model limitations.
- **Data Splitting**: Implement multiple dev sets for better evaluation and tuning.
- **Additional Datasets**: Extend the framework to handle datasets like CIFAR-10, Fashion-MNIST, or custom datasets.
- **Hyperparameter Tuning**: Integrate automated hyperparameter tuning methods (e.g., grid search, random search).

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.


