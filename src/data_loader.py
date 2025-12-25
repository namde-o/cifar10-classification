"""
data_loader.py - Dataset Loading Module

Purpose: Load CIFAR-10 dataset and create train/validation/test splits
Time Complexity: O(n) where n is number of samples
Space Complexity: O(n) to store dataset in memory
"""

import numpy as np
from tensorflow import keras
from typing import Tuple

def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset from Keras datasets.
    
    Returns:
        (x_train, y_train), (x_test, y_test)
        - x_train: (50000, 32, 32, 3) training images
        - y_train: (50000,) training labels (0-9)
        - x_test: (10000, 32, 32, 3) test images
        - y_test: (10000,) test labels (0-9)
    
    Why Keras datasets?
    - Pre-downloaded and verified
    - Standard benchmark splits
    - Production-ready format
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Flatten labels from (n, 1) to (n,)
    # Common beginner mistake: forgetting this causes shape mismatches
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return (x_train, y_train), (x_test, y_test)

def create_validation_split(x_train: np.ndarray, y_train: np.ndarray, 
                           val_split: float = 0.2) -> Tuple:
    """
    Split training data into train and validation sets.
    
    Args:
        x_train: Training images
        y_train: Training labels
        val_split: Fraction of data for validation (default: 0.2 = 20%)
    
    Returns:
        x_train_new, y_train_new, x_val, y_val
    
    Why validation set?
    - Monitor overfitting during training
    - Tune hyperparameters without touching test set
    - Test set should ONLY be used at the very end
    
    Time Complexity: O(1) - just array slicing
    Space Complexity: O(1) - creates views, not copies
    """
    val_size = int(len(x_train) * val_split)
    
    # Use last val_size samples for validation
    # Random shuffle happens in preprocessing
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train_new = x_train[:-val_size]
    y_train_new = y_train[:-val_size]
    
    print(f"Training samples: {len(x_train_new)}")
    print(f"Validation samples: {len(x_val)}")
    
    return x_train_new, y_train_new, x_val, y_val

def get_class_names() -> list:
    """
    Return CIFAR-10 class names for visualization.
    
    Returns:
        List of 10 class names
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']