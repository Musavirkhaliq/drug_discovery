#!/usr/bin/env python3
# Property Prediction with DeepChem

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

# Import our custom modules
import sys
sys.path.append('.')
from models.property_prediction import build_random_forest_model, build_graph_conv_model

# Create output directory for plots
os.makedirs('output', exist_ok=True)

print("Starting property prediction with DeepChem...")

# Load Tox21 dataset with ECFP fingerprints
print("Loading Tox21 dataset with ECFP fingerprints...")
tasks, datasets, transformers = load_tox21(featurizer='ECFP', split='scaffold')
train_dataset, valid_dataset, test_dataset = datasets

print(f"Number of tasks: {len(tasks)}")
print(f"Tasks: {tasks}")
print(f"Number of compounds in train set: {train_dataset.X.shape[0]}")
print(f"Number of compounds in validation set: {valid_dataset.X.shape[0]}")
print(f"Number of compounds in test set: {test_dataset.X.shape[0]}")

# Build a Random Forest model
print("\nBuilding and training Random Forest model...")
rf_model = dc.models.SklearnModel(
    model=RandomForestClassifier(
        n_estimators=100, 
        class_weight="balanced", 
        n_jobs=-1
    ),
    model_dir='output/rf_model'
)

# Train the model
rf_model.fit(train_dataset)
print("Random Forest model training complete!")

# Evaluate the model on the test set
print("\nEvaluating Random Forest model on test set...")
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
test_scores = rf_model.evaluate(test_dataset, [metric])
print(f"Mean ROC-AUC: {test_scores['mean-roc_auc_score']:.4f}")

# Make predictions
y_pred = rf_model.predict(test_dataset)

# Calculate ROC-AUC for each task
task_scores = {}
for i, task in enumerate(tasks):
    # Get valid indices (not NaN)
    valid = ~np.isnan(test_dataset.y[:, i])
    if np.sum(valid) == 0:
        continue
    y_true = test_dataset.y[valid, i]
    y_score = y_pred[valid, i]
    
    # Calculate ROC-AUC
    try:
        score = dc.metrics.roc_auc_score(y_true, y_score)
        task_scores[task] = score
        print(f"{task}: ROC-AUC = {score:.4f}")
    except:
        print(f"{task}: Could not calculate ROC-AUC")

# Save RF results to a file
rf_results = pd.DataFrame({
    'Task': list(task_scores.keys()),
    'ROC-AUC': list(task_scores.values())
})
rf_results.to_csv('output/rf_results.csv', index=False)

# Try a simpler model for demonstration
print("\nTraining a simple MultitaskClassifier model...")
n_features = train_dataset.X.shape[1]
dnn_model = dc.models.MultitaskClassifier(
    n_tasks=len(tasks),
    n_features=n_features,
    layer_sizes=[1000, 500],
    dropouts=[0.25, 0.25],
    learning_rate=0.001,
    batch_size=50,
    model_dir='output/dnn_model'
)

# Train for a few epochs
print("Training DNN model...")
losses = []
for i in range(5):  # Train for 5 epochs to save time
    loss = dnn_model.fit(train_dataset, nb_epoch=1)
    losses.append(loss)
    print(f"Epoch {i+1}/5, Loss: {loss}")
print("DNN model training complete!")

# Plot the training curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DNN Training Loss')
plt.grid(True)
plt.savefig('output/dnn_training_loss.png')
print("Saved DNN training loss plot to output/dnn_training_loss.png")

# Evaluate the DNN model
print("\nEvaluating DNN model on test set...")
dnn_test_scores = dnn_model.evaluate(test_dataset, [metric])
print(f"Mean ROC-AUC: {dnn_test_scores['mean-roc_auc_score']:.4f}")

# Make predictions with DNN model
dnn_y_pred = dnn_model.predict(test_dataset)

# Calculate ROC-AUC for each task
dnn_task_scores = {}
for i, task in enumerate(tasks):
    # Get valid indices (not NaN)
    valid = ~np.isnan(test_dataset.y[:, i])
    if np.sum(valid) == 0:
        continue
    y_true = test_dataset.y[valid, i]
    y_score = dnn_y_pred[valid, i]
    
    # Calculate ROC-AUC
    try:
        score = dc.metrics.roc_auc_score(y_true, y_score)
        dnn_task_scores[task] = score
        print(f"{task}: ROC-AUC = {score:.4f}")
    except:
        print(f"{task}: Could not calculate ROC-AUC")

# Save DNN results to a file
dnn_results = pd.DataFrame({
    'Task': list(dnn_task_scores.keys()),
    'ROC-AUC': list(dnn_task_scores.values())
})
dnn_results.to_csv('output/dnn_results.csv', index=False)

# Compare model performance
comparison_df = pd.DataFrame({
    'Task': list(task_scores.keys()),
    'Random Forest': [task_scores.get(task, np.nan) for task in task_scores.keys()],
    'DNN': [dnn_task_scores.get(task, np.nan) for task in task_scores.keys()]
})

# Save comparison to a file
comparison_df.to_csv('output/model_comparison.csv', index=False)

print("\nModel comparison saved to output/model_comparison.csv")
print("Property prediction completed successfully!") 