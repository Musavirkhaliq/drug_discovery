import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, 
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_regression_model(model, dataset, tasks):
    """
    Evaluate a regression model on a dataset.
    
    Parameters
    ----------
    model : dc.models.Model
        DeepChem model to evaluate.
    dataset : dc.data.Dataset
        Dataset to evaluate on.
    tasks : list
        List of task names.
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(dataset)
    y_true = dataset.y
    
    results = {}
    for i, task in enumerate(tasks):
        # Skip nan values
        valid_indices = ~np.isnan(y_true[:, i])
        if sum(valid_indices) == 0:
            logger.warning(f"No valid samples for task {task}")
            continue
            
        y_t = y_true[valid_indices, i]
        y_p = y_pred[valid_indices, i]
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        r2 = r2_score(y_t, y_p)
        mae = mean_absolute_error(y_t, y_p)
        
        results[task] = {
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae
        }
        
        logger.info(f"Task {task}: RMSE = {rmse:.4f}, R² = {r2:.4f}, MAE = {mae:.4f}")
    
    # Calculate average metrics across all tasks
    avg_rmse = np.mean([results[task]['RMSE'] for task in results])
    avg_r2 = np.mean([results[task]['R²'] for task in results])
    avg_mae = np.mean([results[task]['MAE'] for task in results])
    
    results['average'] = {
        'RMSE': avg_rmse,
        'R²': avg_r2,
        'MAE': avg_mae
    }
    
    logger.info(f"Average: RMSE = {avg_rmse:.4f}, R² = {avg_r2:.4f}, MAE = {avg_mae:.4f}")
    
    return results

def evaluate_classification_model(model, dataset, tasks, threshold=0.5):
    """
    Evaluate a classification model on a dataset.
    
    Parameters
    ----------
    model : dc.models.Model
        DeepChem model to evaluate.
    dataset : dc.data.Dataset
        Dataset to evaluate on.
    tasks : list
        List of task names.
    threshold : float, optional (default=0.5)
        Threshold for binary classification.
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(dataset)
    y_true = dataset.y
    
    results = {}
    for i, task in enumerate(tasks):
        # Skip nan values
        valid_indices = ~np.isnan(y_true[:, i])
        if sum(valid_indices) == 0:
            logger.warning(f"No valid samples for task {task}")
            continue
            
        y_t = y_true[valid_indices, i]
        y_p = y_pred[valid_indices, i]
        
        # For binary classification
        if len(np.unique(y_t)) <= 2:
            y_pred_binary = (y_p >= threshold).astype(int)
            
            accuracy = accuracy_score(y_t, y_pred_binary)
            precision = precision_score(y_t, y_pred_binary, zero_division=0)
            recall = recall_score(y_t, y_pred_binary, zero_division=0)
            f1 = f1_score(y_t, y_pred_binary, zero_division=0)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_t, y_p)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_t, y_p)
            pr_auc = auc(recall_curve, precision_curve)
            
            results[task] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
                'ROC': (fpr, tpr),
                'PR': (recall_curve, precision_curve)
            }
            
            logger.info(f"Task {task}: Accuracy = {accuracy:.4f}, ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")
        else:
            # For multi-class classification
            logger.warning(f"Multi-class classification not fully implemented for task {task}")
            # Implement multi-class metrics if needed
    
    # Calculate average metrics across all tasks
    avg_accuracy = np.mean([results[task]['Accuracy'] for task in results])
    avg_roc_auc = np.mean([results[task]['ROC-AUC'] for task in results])
    avg_pr_auc = np.mean([results[task]['PR-AUC'] for task in results])
    
    results['average'] = {
        'Accuracy': avg_accuracy,
        'ROC-AUC': avg_roc_auc,
        'PR-AUC': avg_pr_auc
    }
    
    logger.info(f"Average: Accuracy = {avg_accuracy:.4f}, ROC-AUC = {avg_roc_auc:.4f}, PR-AUC = {avg_pr_auc:.4f}")
    
    return results

def plot_roc_curves(results, tasks, save_path=None):
    """
    Plot ROC curves for classification tasks.
    
    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_classification_model.
    tasks : list
        List of task names.
    save_path : str, optional
        Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    
    for task in tasks:
        if task in results and 'ROC' in results[task]:
            fpr, tpr = results[task]['ROC']
            roc_auc = results[task]['ROC-AUC']
            plt.plot(fpr, tpr, lw=2, label=f'{task} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_pr_curves(results, tasks, save_path=None):
    """
    Plot Precision-Recall curves for classification tasks.
    
    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_classification_model.
    tasks : list
        List of task names.
    save_path : str, optional
        Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    
    for task in tasks:
        if task in results and 'PR' in results[task]:
            recall_curve, precision_curve = results[task]['PR']
            pr_auc = results[task]['PR-AUC']
            plt.plot(recall_curve, precision_curve, lw=2, label=f'{task} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_regression_predictions(y_true, y_pred, task_name, save_path=None):
    """
    Plot predicted vs actual values for regression tasks.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True values.
    y_pred : numpy.ndarray
        Predicted values.
    task_name : str
        Name of the task.
    save_path : str, optional
        Path to save the plot.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.title(f'{task_name}\nRMSE = {rmse:.4f}, R² = {r2:.4f}')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, task_name, threshold=0.5, save_path=None):
    """
    Plot confusion matrix for binary classification tasks.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True values.
    y_pred : numpy.ndarray
        Predicted probabilities.
    task_name : str
        Name of the task.
    threshold : float, optional (default=0.5)
        Threshold for binary classification.
    save_path : str, optional
        Path to save the plot.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 