import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics, including ROC-AUC and Gini coefficient.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    gini = 2 * roc_auc - 1
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Gini Coefficient': gini
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Plot and save the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion Matrix plot saved to {plot_path}")

def plot_curves(y_true, y_prob, output_dir):
    """
    Plot and save ROC Curve and Precision-Recall Curve.
    """
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    gini_score = 2 * auc_score - 1
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f}, Gini = {gini_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_curves.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Evaluation curves plot saved to {plot_path}")

def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot and save feature importances if tree-based, or model coefficients if linear.
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Plot top 15 features or less
        top_n = min(15, len(sorted_features))
        sns.barplot(x=sorted_importances[:top_n], y=sorted_features[:top_n], hue=sorted_features[:top_n], palette='viridis', legend=False)
        plt.title('Random Forest Feature Importance (Top Features)')
        plt.xlabel('Importance Value')
        
    elif hasattr(model, 'coef_'):
        # For Logistic Regression, plot coefficients
        coefs = model.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_coefs = coefs[indices]
        
        top_n = min(15, len(sorted_features))
        sns.barplot(x=sorted_coefs[:top_n], y=sorted_features[:top_n], hue=sorted_features[:top_n], palette='coolwarm', legend=False)
        plt.title('Logistic Regression Coefficients (Top Magnitude)')
        plt.xlabel('Coefficient Value')
    else:
        print("Model does not support feature_importances_ or coef_ for plotting.")
        return
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")

def evaluate_model(model, X_test, y_test, output_dir):
    """
    Perform full model evaluation: print metrics and save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict classes and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Print metrics report
    print("\n" + "="*40)
    print("           MODEL EVALUATION METRICS           ")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:20s} : {v:.4f}")
    print("="*40)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))
    
    # Save metrics to text file
    metrics_path = os.path.join(output_dir, 'metrics_report.txt')
    with open(metrics_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write("           MODEL EVALUATION METRICS           \n")
        f.write("="*40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:20s} : {v:.4f}\n")
        f.write("="*40 + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))
    print(f"Metrics text report saved to {metrics_path}")
    
    # Generate and save plots
    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_curves(y_test, y_prob, output_dir)
    plot_feature_importance(model, list(X_test.columns), output_dir)
    
    return metrics
