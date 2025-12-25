"""
train.py - Model Training Pipeline

Critical section: This is where the magic happens!
"""

import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

def create_callbacks(model_save_path='models/best_model.keras'):
    """
    Create training callbacks for better model management.
    
    Callbacks:
    1. ModelCheckpoint - Save best model based on validation accuracy
    2. EarlyStopping - Stop if validation loss doesn't improve
    3. ReduceLROnPlateau - Reduce learning rate when stuck
    
    Why callbacks?
    - Prevent overfitting (early stopping)
    - Save best model automatically
    - Adaptive learning rate adjustment
    
    For interviews: Shows you understand production ML practices
    
    Time Complexity: O(1) per epoch overhead
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',  # Track validation accuracy
        save_best_only=True,     # Only save when validation improves
        mode='max',              # Maximize validation accuracy
        verbose=1
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,              # Wait 10 epochs before stopping
        restore_best_weights=True,  # Restore weights from best epoch
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,              # Reduce LR by half
        patience=5,              # Wait 5 epochs before reducing
        min_lr=1e-7,
        verbose=1
    )
    
    # Optional: TensorBoard for visualization
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1  # Log weight histograms each epoch
    )
    
    return [checkpoint, early_stop, reduce_lr]

def train_model(model, x_train, y_train, x_val, y_val, 
                epochs=50, batch_size=64, augmentation=None):
    """
    Train the model with optional data augmentation.
    
    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_val, y_val: Validation data
        epochs: Maximum number of training epochs
        batch_size: Number of samples per gradient update
        augmentation: ImageDataGenerator for data augmentation
    
    Training process:
    1. Forward pass: Compute predictions
    2. Calculate loss
    3. Backward pass: Compute gradients
    4. Update weights using optimizer
    5. Repeat for each batch
    
    Time Complexity: O(epochs * n/batch_size * p)
        where n=samples, p=parameters
    
    Space Complexity: O(batch_size * input_size) for batch processing
    
    Common beginner mistakes:
    - Using test set for validation (NEVER do this!)
    - Not shuffling training data
    - Batch size too large (GPU memory) or too small (noisy gradients)
    """
    callbacks = create_callbacks()
    
    if augmentation:
        # Training with data augmentation
        print("Training with data augmentation...")
        history = model.fit(
            augmentation.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            steps_per_epoch=len(x_train) // batch_size,
            verbose=1
        )
    else:
        # Standard training
        print("Training without augmentation...")
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    return history

def plot_training_history(history, save_path='results/training_history.png'):
    """
    Visualize training and validation metrics.
    
    Why visualize?
    - Diagnose overfitting (train acc >> val acc)
    - Check if model is still learning
    - Identify when to stop training
    
    For resume: Include this plot in your GitHub README
    Shows you understand ML evaluation beyond just numbers
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    # Check for overfitting
    if final_train_acc - final_val_acc > 0.1:
        print("⚠️  Warning: Significant overfitting detected!")
        print("   Consider: More dropout, data augmentation, or regularization")

def save_model(model, path='models/final_model.keras'):
    """
    Save trained model for deployment.
    
    Keras SavedModel format advantages:
    - Contains architecture, weights, and training config
    - Can be deployed to TensorFlow Serving
    - Interoperable with TensorFlow.js and TFLite
    
    Time Complexity: O(p) where p = parameters
    Space Complexity: Model size on disk
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")
    
    # Also save just weights (lighter format)
    weights_path = path.replace('.keras', '_weights.h5')
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")