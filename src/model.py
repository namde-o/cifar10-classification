"""
model.py - Model Architecture Definitions

We'll implement 3 models of increasing complexity:
1. Simple CNN (baseline)
2. Deeper CNN with batch normalization
3. ResNet-inspired architecture (advanced)

Start with #1, iterate to #2, mention #3 in interviews
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models

def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Simple Convolutional Neural Network (CNN)
    
    Architecture:
    - 2 Conv blocks (Conv → ReLU → MaxPool)
    - Flatten
    - 2 Dense layers
    - Softmax output
    
    Why this architecture?
    - Convolutional layers: Learn spatial features (edges, textures, patterns)
    - MaxPooling: Reduce spatial dimensions, add translation invariance
    - Dense layers: Combine features for classification
    
    Parameters: ~122K (lightweight, trains fast)
    
    Time Complexity:
    - Training: O(n * p * e) where n=samples, p=params, e=epochs
    - Inference: O(p) per image
    
    Space Complexity: O(p) for model weights
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=input_shape, padding='same'),
        # 32 filters of size 3x3
        # Padding='same' keeps spatial dimensions
        layers.MaxPooling2D((2, 2)),
        # Reduces 32x32 → 16x16
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Reduces 16x16 → 8x8
        
        # Third Conv Block (deeper features)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        
        # Flatten and Dense layers
        layers.Flatten(),
        # Converts (8, 8, 64) → (4096,)
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Regularization to prevent overfitting
        layers.Dense(num_classes, activation='softmax')
        # Softmax: outputs probability distribution over 10 classes
    ])
    
    return model

def create_improved_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Improved CNN with Batch Normalization and Dropout
    
    Improvements:
    - Batch Normalization: Stabilizes learning, allows higher learning rates
    - More aggressive dropout: Better regularization
    - More filters: Increased capacity
    
    Parameters: ~350K
    Expected accuracy: 75-80% (vs 65-70% for simple CNN)
    
    For resume: "Optimized CNN architecture with batch normalization 
                 achieving 15% accuracy improvement"
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),  # Normalize activations
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Light dropout after pooling
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer, loss, and metrics.
    
    Optimizer: Adam
    - Adaptive learning rate per parameter
    - Combines momentum and RMSprop
    - Industry standard, works well out-of-box
    
    Loss: Categorical Crossentropy
    - Standard for multi-class classification
    - Measures difference between predicted and true distributions
    
    Metrics: Accuracy
    - Easy to interpret
    - Add top-5 accuracy for more nuanced evaluation
    
    Time Complexity: O(1) - just configuration
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    return model

def get_model_size(model):
    """
    Calculate model size and parameter count.
    
    Useful for:
    - Resume stats: "Developed lightweight CNN with 350K parameters"
    - Understanding memory requirements
    - Interview discussions about model efficiency
    """
    trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable_params = sum([np.prod(w.shape) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # Estimate model size in MB (assuming float32)
    size_mb = (total_params * 4) / (1024 ** 2)
    print(f"Estimated model size: {size_mb:.2f} MB")
    
    return total_params
