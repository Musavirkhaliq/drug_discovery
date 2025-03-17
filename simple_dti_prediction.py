#!/usr/bin/env python3
# Simple Drug-Target Interaction Prediction with DeepChem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import deepchem as dc
from deepchem.molnet import load_bace_classification
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import tensorflow as tf

# Create output directory for plots
os.makedirs('output', exist_ok=True)

print("Starting simple drug-target interaction prediction with DeepChem...")

# Load BACE dataset (inhibition of human β-secretase 1)
print("Loading BACE dataset...")
tasks, datasets, transformers = load_bace_classification(splitter='scaffold')
train_dataset, valid_dataset, test_dataset = datasets

print(f"Number of tasks: {len(tasks)}")
print(f"Tasks: {tasks}")
print(f"Number of compounds in train set: {train_dataset.X.shape[0]}")
print(f"Number of compounds in validation set: {valid_dataset.X.shape[0]}")
print(f"Number of compounds in test set: {test_dataset.X.shape[0]}")

# Extract features and labels
train_X = train_dataset.X
train_y = train_dataset.y
valid_X = valid_dataset.X
valid_y = valid_dataset.y
test_X = test_dataset.X
test_y = test_dataset.y

# Get SMILES strings from IDs
train_smiles = train_dataset.ids
valid_smiles = valid_dataset.ids
test_smiles = test_dataset.ids

print(f"Number of compounds: {len(train_smiles) + len(valid_smiles) + len(test_smiles)}")

# Take a subset of the data for faster processing
max_samples = 500
if len(train_X) > max_samples:
    print(f"\nTaking a subset of {max_samples} samples for faster processing...")
    indices = np.random.choice(len(train_X), max_samples, replace=False)
    train_X = train_X[indices]
    train_y = train_y[indices]
    train_smiles = [train_smiles[i] for i in indices]

if len(valid_X) > max_samples // 4:
    indices = np.random.choice(len(valid_X), max_samples // 4, replace=False)
    valid_X = valid_X[indices]
    valid_y = valid_y[indices]
    valid_smiles = [valid_smiles[i] for i in indices]

if len(test_X) > max_samples // 4:
    indices = np.random.choice(len(test_X), max_samples // 4, replace=False)
    test_X = test_X[indices]
    test_y = test_y[indices]
    test_smiles = [test_smiles[i] for i in indices]

# Featurize molecules using Morgan fingerprints
print("\nFeaturizing molecules using Morgan fingerprints...")
def generate_morgan_fingerprints(smiles_list, radius=2, nBits=1024):
    """Generate Morgan fingerprints for a list of SMILES strings."""
    fingerprints = []
    valid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fingerprints.append(np.array(fp, dtype=np.float32))  # Convert to float32
            valid_smiles.append(smiles)
        else:
            print(f"Warning: Could not convert SMILES to molecule: {smiles}")
    
    return np.array(fingerprints, dtype=np.float32), valid_smiles  # Ensure float32 type

# Generate fingerprints
train_fps, valid_train_smiles = generate_morgan_fingerprints(train_smiles)
valid_fps, valid_valid_smiles = generate_morgan_fingerprints(valid_smiles)
test_fps, valid_test_smiles = generate_morgan_fingerprints(test_smiles)

# Create datasets with the fingerprints
train_dataset_fp = dc.data.NumpyDataset(X=train_fps, y=train_y.astype(np.float32))  # Convert to float32
valid_dataset_fp = dc.data.NumpyDataset(X=valid_fps, y=valid_y.astype(np.float32))  # Convert to float32
test_dataset_fp = dc.data.NumpyDataset(X=test_fps, y=test_y.astype(np.float32))  # Convert to float32

# Print data types for debugging
print(f"Feature data type: {train_fps.dtype}")
print(f"Label data type: {train_dataset_fp.y.dtype}")

# Build a simple DNN model for classification
print("\nBuilding and training a DNN model for BACE inhibition prediction...")
input_dim = train_fps.shape[1]
dnn_model = dc.models.MultitaskClassifier(
    n_tasks=len(tasks),
    n_features=input_dim,
    layer_sizes=[256, 128, 64],
    dropouts=[0.2, 0.2, 0.2],
    learning_rate=0.001,
    batch_size=32,
    model_dir='output/bace_model_simple'
)

# Train the model
print("Training DNN model...")
losses = []
for i in range(20):  # Train for 20 epochs
    loss = dnn_model.fit(train_dataset_fp, nb_epoch=1)
    losses.append(loss)
    print(f"Epoch {i+1}/20, Loss: {loss}")
print("DNN model training complete!")

# Plot the training curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DNN Training Loss')
plt.grid(True)
plt.savefig('output/bace_training_loss.png')
print("Saved DNN training loss plot to output/bace_training_loss.png")

# Evaluate the model on the test set
print("\nEvaluating DNN model on test set...")
metric_auc = dc.metrics.Metric(dc.metrics.roc_auc_score)
metric_acc = dc.metrics.Metric(dc.metrics.accuracy_score)
test_scores = dnn_model.evaluate(test_dataset_fp, [metric_auc, metric_acc])
print(f"Test ROC-AUC: {test_scores['roc_auc_score']:.4f}")
print(f"Test Accuracy: {test_scores['accuracy_score']:.4f}")

# Make predictions on the test set
print("\nMaking predictions on test set...")
y_pred = dnn_model.predict(test_dataset_fp)

# For binary classification, extract the positive class probability
y_pred_proba = y_pred[:, 0, 1] if y_pred.shape[2] == 2 else y_pred[:, 0]
y_pred_class = (y_pred_proba > 0.5).astype(int)

# Calculate ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(test_y[:, 0], y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for BACE Inhibition Prediction')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('output/bace_roc_curve.png')
print("Saved ROC curve to output/bace_roc_curve.png")

# Save results to a file
results = pd.DataFrame({
    'SMILES': valid_test_smiles,
    'Actual': test_y[:, 0],
    'Predicted_Probability': y_pred_proba,
    'Predicted_Class': y_pred_class
})
results.to_csv('output/bace_prediction_results.csv', index=False)

print("\nResults saved to output/bace_prediction_results.csv")
print("Simple BACE inhibition prediction completed successfully!") 