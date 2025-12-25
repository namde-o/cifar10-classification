"""
main.py - End-to-End Image Classification Pipeline

Run this script to execute the complete workflow:
1. Load and preprocess data
2. Build and train model
3. Evaluate performance
4. Generate visualizations

Usage:
    python main.py --model simple    # Train simple CNN
    python main.py --model improved  # Train improved CNN
    python main.py --augment         # Enable data augmentation
"""

import argparse
import numpy as np
import tensorflow as tf

# Set random seeds for reproducibility
# Critical for scientific experiments and debugging
np.random.seed(42)
tf.random.set_seed(42)

# Import our custom modules
from src.data_loader import load_cifar10, create_validation_split, get_class_names
from src.preprocessing import (normalize_images, preprocess_labels, 
                               augment_data, compute_dataset_statistics)
from src.model import create_simple_cnn, create_improved_cnn, compile_model, get_model_size
from src.train import train_model, plot_training_history, save_model
from src.evaluate import (evaluate_model, plot_confusion_matrix, 
                         visualize_predictions, analyze_errors,
                         generate_classification_report)

def main(args):
    """
    Main execution pipeline.
    
    This orchestrates the entire workflow from raw data to trained model.
    
    Time Complexity Summary:
    - Data loading: O(n)
    - Preprocessing: O(n)
    - Training: O(epochs * n * p / batch_size)
    - Evaluation: O(n * p)
    Total: Dominated by training complexity
    
    Space Complexity: O(n + p)
    - n: dataset size in memory
    - p: model parameters
    """
    
    print("="*70)
    print(" "*15 + "CIFAR-10 IMAGE CLASSIFICATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data augmentation: {args.augment}")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("STEP 1: Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    class_names = get_class_names()
    
    print(f"Training set: {x_train.shape[0]} images")
    print(f"Test set: {x_test.shape[0]} images")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Classes: {class_names}\n")
    
    # Step 2: Create validation split
    print("STEP 2: Creating validation split...")
    x_train, y_train, x_val, y_val = create_validation_split(x_train, y_train)
    print()
    
    # Step 3: Preprocess data
    print("STEP 3: Preprocessing data...")
    
    # Normalize pixel values
    x_train, x_val, x_test = normalize_images(x_train, x_val, x_test)
    
    # Compute statistics (optional, for analysis)
    compute_dataset_statistics(x_train)
    
    # One-hot encode labels
    y_train, y_val, y_test = preprocess_labels(y_train, y_val, y_test)
    
    # Setup data augmentation if requested
    datagen = None
    if args.augment:
        datagen = augment_data(x_train, y_train)
    
    print()
    
    # Step 4: Build model
    print("STEP 4: Building model architecture...")
    
    if args.model == 'simple':
        model = create_simple_cnn()
    elif args.model == 'improved':
        model = create_improved_cnn()
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Compile model
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # Display model size
    get_model_size(model)
    print()
    
    # Step 5: Train model
    print("STEP 5: Training model...")
    print("This may take several minutes...")
    
    history = train_model(
        model, 
        x_train, y_train, 
        x_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augmentation=datagen
    )
    
    # Plot training history
    plot_training_history(history)
    print()
    
    # step 6 Eavaluate model
    print("STEP 6: Evaluating model on test set...")
    
    # Comprehensive evaluation
    y_true, y_pred, test_accuracy = evaluate_model(model, x_test, y_test, class_names)
    
    # Generate confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Visualize sample predictions
    visualize_predictions(model, x_test, y_test, class_names)
    
    # Error analysis
    analyze_errors(model, x_test, y_test, class_names)
    
    # Detailed classification report
    generate_classification_report(y_true, y_pred, class_names)
    
    # Step 7: Save the trained model
    print("\nSTEP 7: Saving model...")
    save_model(model)
    
    # Summary of results
    print("\n" + "="*70)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"\nResults saved to:")
    print("  - models/best_model.keras")
    print("  - results/training_history.png")
    print("  - results/confusion_matrix.png")
    print("  - results/sample_predictions.png")
    print("\nNext steps:")
    print("  1. Review visualizations in results/ folder")
    print("  2. Analyze confusion matrix for improvement opportunities")
    print("  3. Try hyperparameter tuning (learning rate, batch size)")
    print("  4. Experiment with different architectures")
    print("  5. Push to GitHub and update README with results")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classification')
    
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'improved'],
                       help='Model architecture to use')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    
    args = parser.parse_args()
    
    main(args)