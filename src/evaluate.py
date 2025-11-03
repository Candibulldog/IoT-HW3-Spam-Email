"""
Model evaluation module for spam detection.

Provides comprehensive metrics and visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
    }
    
    return metrics


def plot_confusion_matrix(model, X_test, y_test, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        save_path: Optional path to save figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Ham', 'Spam'])
    plt.yticks([0.5, 1.5], ['Ham', 'Spam'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(model, X_test, y_test, save_path: str = None):
    """
    Plot ROC curve.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        save_path: Optional path to save figure
    """
    if not hasattr(model, 'predict_proba'):
        print("Model does not support probability predictions. Skipping ROC curve.")
        return
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def print_classification_report(model, X_test, y_test):
    """
    Print detailed classification report.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    print("\nClassification Report:")
    print("="*50)
    print(report)


def compare_models(models: dict, X_test, y_test) -> None:
    """
    Compare multiple models side-by-side.
    
    Args:
        models: Dictionary of {model_name: model_object}
        X_test: Test features
        y_test: Test labels
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        results.append(metrics)
    
    # Create comparison table
    print("\nModel Comparison")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model']:<20} "
              f"{result['accuracy']:<12.4f} "
              f"{result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} "
              f"{result['f1_score']:<12.4f}")
    
    print("="*80)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    model_names = [r['model'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    axes[0].bar(model_names, accuracies, color='skyblue')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim(0.8, 1.0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # F1-Score comparison
    f1_scores = [r['f1_score'] for r in results]
    axes[1].bar(model_names, f1_scores, color='lightcoral')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('Model F1-Score Comparison')
    axes[1].set_ylim(0.8, 1.0)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import MinMaxScaler
    
    # Generate dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                               n_redundant=5, random_state=42)
    
    # Make data non-negative for MultinomialNB
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train simple model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Testing evaluation functions...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nMetrics: {metrics}")
    
    print_classification_report(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
