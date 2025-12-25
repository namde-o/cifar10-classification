"""
evaluate.py - Model Evaluation and Analysis

Critical for interviews: Being able to explain your metrics!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import os

def evaluate_model(model, x_test, y_test, class_names):
    """
    Comprehensive model evaluation on test set.
    
    Metrics:
    1. Accuracy: Overall correctness
    2. Precision: Of predicted positives, how many are correct?
    3. Recall: Of actual positives, how many did we find?
    4. F1-Score: Harmonic mean of precision and recall
    
    Time Complexity: O(n * p) for inference on n test samples
    Space Complexity: O(n * c) for predictions, c = num_classes
    
    CRITICAL: Only evaluate on test set ONCE at the very end!
    """
    print("="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)
    
    # Get predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate overall accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    print("\nPer-Class Performance:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    
    # Macro and weighted averages
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    print("-" * 60)
    print(f"{'Macro Avg':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    
    return y_true, y_pred, test_accuracy

def plot_confusion_matrix(y_true, y_pred, class_names, 
                         save_path='results/confusion_matrix.png'):
    """
    Create confusion matrix visualization.
    
    Confusion Matrix shows:
    - Diagonal: Correct predictions
    - Off-diagonal: Misclassifications
    - Which classes are confused with each other
    
    Interview tip: Be ready to explain what patterns you see!
    Example: "Cars and trucks are often confused due to similar shapes"
    
    Time Complexity: O(n) to compute, O(c²) to visualize
    Space Complexity: O(c²) matrix
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix (optional, shows percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_path}")
    
    # Identify most confused classes
    print("\nMost Common Misclassifications:")
    print("-" * 60)
    
    # Get off-diagonal elements
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # Remove correct predictions
    
    # Find top 5 misclassifications
    top_errors = []
    for _ in range(min(5, cm_copy.size)):
        true_idx, pred_idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        count = cm_copy[true_idx, pred_idx]
        if count > 0:
            top_errors.append((class_names[true_idx], class_names[pred_idx], count))
            cm_copy[true_idx, pred_idx] = 0
    
    for i, (true_class, pred_class, count) in enumerate(top_errors, 1):
        print(f"{i}. {true_class} → {pred_class}: {count} errors")

def visualize_predictions(model, x_test, y_test, class_names, num_samples=10,
                         save_path='results/sample_predictions.png'):
    """
    Visualize sample predictions with confidence scores.
    
    Great for:
    - GitHub README showcase
    - Understanding where model fails
    - Interview demonstrations
    
    Shows both correct and incorrect predictions with confidence levels
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # Make predictions
    y_pred_probs = model.predict(x_test[indices], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test[indices], axis=1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Display image
        ax.imshow(x_test[idx])
        ax.axis('off')
        
        # Get prediction info
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        confidence = y_pred_probs[i][y_pred[i]]
        
        # Color code: green if correct, red if wrong
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        
        # Title with prediction
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}"
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions saved to {save_path}")

def analyze_errors(model, x_test, y_test, class_names, 
                  save_path='results/error_analysis.png'):
    """
    Deep dive into model errors.
    
    Analysis:
    - Most confident mistakes (high confidence but wrong)
    - Least confident correct predictions (low confidence but right)
    
    For interviews: Shows critical thinking about model behavior
    """
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Get confidence scores
    confidence_scores = np.max(y_pred_probs, axis=1)
    
    # Find errors
    errors = y_pred != y_true
    
    # Most confident errors
    confident_errors = np.where(errors)[0]
    if len(confident_errors) > 0:
        sorted_errors = confident_errors[np.argsort(-confidence_scores[confident_errors])]
        
        print("\n" + "="*60)
        print("MOST CONFIDENT MISTAKES (Model was wrong but very sure)")
        print("="*60)
        
        for i, idx in enumerate(sorted_errors[:5], 1):
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            conf = confidence_scores[idx]
            print(f"{i}. True: {true_label:<12} | Predicted: {pred_label:<12} | Confidence: {conf:.2%}")
        
        print("\nAction items:")
        print("- Review these misclassified samples")
        print("- Consider adding similar examples to training set")
        print("- Check if certain classes need better representation")

def generate_classification_report(y_true, y_pred, class_names):
    """
    Generate detailed sklearn classification report.
    
    Useful for:
    - Technical documentation
    - Comparing multiple models
    - Identifying weak classes
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names, 
                                   digits=4)
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    return report