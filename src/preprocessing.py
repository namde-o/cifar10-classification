"""
preprocessing.py - Data Preprocessing Module

Purpose: Normalize and prepare data for model training
Critical for neural network performance!
"""

import numpy as np
from sklearn.model_selection import train_test_split

def normalize_images(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
    """
    Normalize pixel values from [0, 255] to [0, 1].
    
    Why normalize?
    1. Neural networks converge faster with normalized inputs
    2. Prevents gradient explosion/vanishing
    3. Makes learning rate tuning easier
    4. Different features (R, G, B channels) on same scale
    
    Time Complexity: O(n) - single pass through data
    Space Complexity: O(1) - in-place division (converts to float)
    
    Common beginner mistake: Normalizing train but forgetting test set!
    """
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Normalized pixel range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    
    return x_train, x_val, x_test

def preprocess_labels(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, 
                     num_classes: int = 10):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Example: 
        Label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    
    Why one-hot encoding?
    - Neural networks output probability distributions
    - Enables use of cross-entropy loss (gold standard for classification)
    - Treats classes as categorical (no implicit ordering)
    
    Alternative: Use sparse categorical crossentropy and skip this step
    (I'll show both approaches)
    
    Time Complexity: O(n * k) where k = num_classes
    Space Complexity: O(n * k) - stores full one-hot matrix
    """
    from tensorflow import keras
    
    y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
    y_val_encoded = keras.utils.to_categorical(y_val, num_classes)
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Label shape before: {y_train.shape}")
    print(f"Label shape after: {y_train_encoded.shape}")
    
    return y_train_encoded, y_val_encoded, y_test_encoded

def augment_data(x_train: np.ndarray, y_train: np.ndarray):
    """
    Apply data augmentation to increase training data diversity.
    
    Techniques:
    - Random horizontal flips
    - Random rotations (±15 degrees)
    - Random shifts
    - Random zoom
    
    Why augmentation?
    - Prevents overfitting (acts as regularization)
    - Teaches model to be invariant to transformations
    - Increases effective dataset size without collecting more data
    
    Time Complexity: O(1) setup, O(n) per epoch during training
    Space Complexity: O(1) - generates on-the-fly
    
    For resume: "Implemented data augmentation to improve model generalization"
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=15,           # Rotate images up to 15 degrees
        width_shift_range=0.1,       # Shift horizontally by 10%
        height_shift_range=0.1,      # Shift vertically by 10%
        horizontal_flip=True,        # Flip images horizontally
        fill_mode='nearest'          # Fill empty pixels after transformation
    )
    
    datagen.fit(x_train)
    
    print("Data augmentation configured")
    print("- Rotation: ±15°")
    print("- Shifts: ±10%")
    print("- Horizontal flip: enabled")
    
    return datagen

def compute_dataset_statistics(x_train: np.ndarray):
    """
    Compute mean and std for potential standardization.
    
    Standardization: (x - mean) / std → zero mean, unit variance
    
    When to use:
    - Transfer learning (match ImageNet preprocessing)
    - Very deep networks
    
    For CIFAR-10 from scratch: normalization to [0,1] usually sufficient
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    mean = np.mean(x_train, axis=(0, 1, 2))
    std = np.std(x_train, axis=(0, 1, 2))
    
    print(f"Dataset mean (RGB): {mean}")
    print(f"Dataset std (RGB): {std}")
    
    return mean, std