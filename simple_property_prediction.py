#!/usr/bin/env python3
# Simple Property Prediction with DeepChem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import deepchem as dc
from deepchem.molnet import load_tox21
from rdkit import Chem
from rdkit.Chem import Draw
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Create output directory for plots
os.makedirs('output', exist_ok=True)

print("Starting simple property prediction with DeepChem...")

# Load Tox21 dataset with ECFP fingerprints
print("Loading Tox21 dataset with ECFP fingerprints...")
tasks, datasets, transformers = load_tox21(featurizer='ECFP', splitter='scaffold')
train_dataset, valid_dataset, test_dataset = datasets

print(f"Number of tasks: {len(tasks)}")
print(f"Tasks: {tasks}")
print(f"Number of compounds in train set: {train_dataset.X.shape[0]}")
print(f"Number of compounds in validation set: {valid_dataset.X.shape[0]}")
print(f"Number of compounds in test set: {test_dataset.X.shape[0]}")

# Select a single task (NR-AR) for simplicity
task_idx = 0
task_name = tasks[task_idx]
print(f"\nFocusing on task: {task_name}")

# Extract data for the selected task
train_X = train_dataset.X
train_y = train_dataset.y[:, task_idx:task_idx+1]
valid_X = valid_dataset.X
valid_y = valid_dataset.y[:, task_idx:task_idx+1]
test_X = test_dataset.X
test_y = test_dataset.y[:, task_idx:task_idx+1]

# Create single-task datasets
train_dataset_single = dc.data.NumpyDataset(X=train_X, y=train_y)
valid_dataset_single = dc.data.NumpyDataset(X=valid_X, y=valid_y)
test_dataset_single = dc.data.NumpyDataset(X=test_X, y=test_y)

# Build a Random Forest model
print("\nBuilding and training Random Forest model...")
rf_model = dc.models.SklearnModel(
    model=RandomForestClassifier(
        n_estimators=100, 
        class_weight="balanced", 
        n_jobs=-1
    ),
    model_dir='output/rf_model_simple'
)

# Train the model
rf_model.fit(train_dataset_single)
print("Random Forest model training complete!")

# Evaluate the model on the test set
print("\nEvaluating Random Forest model on test set...")
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
test_scores = rf_model.evaluate(test_dataset_single, [metric])
print(f"{task_name} ROC-AUC: {test_scores['roc_auc_score']:.4f}")

# Make predictions
y_pred_rf = rf_model.predict(test_dataset_single)

# Try a simple DNN model
print("\nTraining a simple MultitaskClassifier model...")
n_features = train_X.shape[1]
dnn_model = dc.models.MultitaskClassifier(
    n_tasks=1,  # Single task
    n_features=n_features,
    layer_sizes=[500, 200],
    dropouts=[0.25, 0.25],
    learning_rate=0.001,
    batch_size=50,
    model_dir='output/dnn_model_simple'
)

# Train for a few epochs
print("Training DNN model...")
losses = []
for i in range(10):  # Train for 10 epochs
    loss = dnn_model.fit(train_dataset_single, nb_epoch=1)
    losses.append(loss)
    print(f"Epoch {i+1}/10, Loss: {loss}")
print("DNN model training complete!")

# Plot the training curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DNN Training Loss')
plt.grid(True)
plt.savefig('output/dnn_training_loss_simple.png')
print("Saved DNN training loss plot to output/dnn_training_loss_simple.png")

# Evaluate the DNN model
print("\nEvaluating DNN model on test set...")
dnn_test_scores = dnn_model.evaluate(test_dataset_single, [metric])
print(f"{task_name} ROC-AUC: {dnn_test_scores['roc_auc_score']:.4f}")

# Make predictions with DNN model
y_pred_dnn = dnn_model.predict(test_dataset_single)

# Compare model performance
comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'DNN'],
    'ROC-AUC': [test_scores['roc_auc_score'], dnn_test_scores['roc_auc_score']]
})

# Save comparison to a file
comparison_df.to_csv('output/model_comparison_simple.csv', index=False)
print("\nModel comparison saved to output/model_comparison_simple.csv")

# Plot ROC curves
# Get valid indices (not NaN)
valid = ~np.isnan(test_y.flatten())
if np.sum(valid) > 0:
    y_true = test_y[valid].flatten()
    
    # Extract the first column of predictions for valid samples
    # The model outputs predictions for all tasks, but we only want the first one
    y_score_rf = y_pred_rf[valid, 0]
    
    # For DNN, we need to extract the positive class probability (column 1)
    # MultitaskClassifier outputs probabilities for each class
    if y_pred_dnn.shape[1] == 2:  # Binary classification (2 columns: [neg_prob, pos_prob])
        y_score_dnn = y_pred_dnn[valid, 1]  # Use positive class probability
    else:
        y_score_dnn = y_pred_dnn[valid, 0]  # Use the only column available
    
    print(f"Shapes for ROC calculation - y_true: {y_true.shape}, y_score_rf: {y_score_rf.shape}, y_score_dnn: {y_score_dnn.shape}")
    
    try:
        # Calculate ROC curves
        fpr_rf, tpr_rf, _ = roc_curve(y_true, y_score_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        
        fpr_dnn, tpr_dnn, _ = roc_curve(y_true, y_score_dnn)
        roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
        plt.plot(fpr_dnn, tpr_dnn, label=f'DNN (AUC = {roc_auc_dnn:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {task_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig('output/roc_curves_simple.png')
        print("Saved ROC curves to output/roc_curves_simple.png")
    except Exception as e:
        print(f"Error calculating ROC curves: {e}")

print("Simple property prediction completed successfully!") 