#!/usr/bin/env python3
# Drug-Target Interaction Prediction with DeepChem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import deepchem as dc
from deepchem.molnet import load_davis
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import tensorflow as tf

# Create output directory for plots
os.makedirs('output', exist_ok=True)

print("Starting drug-target interaction prediction with DeepChem...")

# Load Davis dataset (drug-target binding affinity)
print("Loading Davis dataset...")
tasks, datasets, transformers = load_davis()
train_dataset, valid_dataset, test_dataset = datasets

print(f"Number of tasks: {len(tasks)}")
print(f"Tasks: {tasks}")
print(f"Number of drug-target pairs in train set: {train_dataset.X.shape[0]}")
print(f"Number of drug-target pairs in validation set: {valid_dataset.X.shape[0]}")
print(f"Number of drug-target pairs in test set: {test_dataset.X.shape[0]}")

# Extract drugs and proteins from the dataset
print("\nExtracting drugs and proteins from the dataset...")
# Davis dataset contains drug SMILES in X[:, 0] and protein sequences in X[:, 1]
drugs_train = train_dataset.X[:, 0]
proteins_train = train_dataset.X[:, 1]
y_train = train_dataset.y

drugs_valid = valid_dataset.X[:, 0]
proteins_valid = valid_dataset.X[:, 1]
y_valid = valid_dataset.y

drugs_test = test_dataset.X[:, 0]
proteins_test = test_dataset.X[:, 1]
y_test = test_dataset.y

print(f"Number of unique drugs: {len(np.unique(drugs_train))}")
print(f"Number of unique proteins: {len(np.unique(proteins_train))}")

# Featurize drugs using Morgan fingerprints
print("\nFeaturizing drugs using Morgan fingerprints...")
def generate_morgan_fingerprints(smiles_list, radius=2, nBits=1024):
    """Generate Morgan fingerprints for a list of SMILES strings."""
    fingerprints = []
    valid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fingerprints.append(np.array(fp))
            valid_smiles.append(smiles)
        else:
            print(f"Warning: Could not convert SMILES to molecule: {smiles}")
    
    return np.array(fingerprints), valid_smiles

drug_features_train, valid_drugs_train = generate_morgan_fingerprints(drugs_train)
drug_features_valid, valid_drugs_valid = generate_morgan_fingerprints(drugs_valid)
drug_features_test, valid_drugs_test = generate_morgan_fingerprints(drugs_test)

# Featurize proteins using amino acid composition
print("Featurizing proteins using amino acid composition...")
def featurize_protein_sequence(sequences):
    """Calculate amino acid composition for protein sequences."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    features = []
    
    for seq in sequences:
        # Count amino acids
        counts = {aa: seq.count(aa) / len(seq) for aa in amino_acids}
        features.append([counts[aa] for aa in amino_acids])
    
    return np.array(features)

protein_features_train = featurize_protein_sequence(proteins_train)
protein_features_valid = featurize_protein_sequence(proteins_valid)
protein_features_test = featurize_protein_sequence(proteins_test)

# Create combined features for a simple model
print("Creating combined features for model input...")
combined_features_train = np.concatenate([drug_features_train, protein_features_train], axis=1)
combined_features_valid = np.concatenate([drug_features_valid, protein_features_valid], axis=1)
combined_features_test = np.concatenate([drug_features_test, protein_features_test], axis=1)

# Create DeepChem datasets with the combined features
train_dataset_combined = dc.data.NumpyDataset(X=combined_features_train, y=y_train)
valid_dataset_combined = dc.data.NumpyDataset(X=combined_features_valid, y=y_valid)
test_dataset_combined = dc.data.NumpyDataset(X=combined_features_test, y=y_test)

# Build a simple DNN model for DTI prediction
print("\nBuilding and training a DNN model for DTI prediction...")
input_dim = combined_features_train.shape[1]
dnn_model = dc.models.MultitaskRegressor(
    n_tasks=1,
    n_features=input_dim,
    layer_sizes=[512, 256, 128],
    dropouts=[0.2, 0.2, 0.2],
    learning_rate=0.001,
    model_dir='output/dti_model'
)

# Train the model
print("Training DNN model...")
losses = []
for i in range(10):  # Train for 10 epochs
    loss = dnn_model.fit(train_dataset_combined, nb_epoch=1)
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
plt.savefig('output/dti_training_loss.png')
print("Saved DNN training loss plot to output/dti_training_loss.png")

# Evaluate the model on the test set
print("\nEvaluating DNN model on test set...")
metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score)
metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)
test_scores = dnn_model.evaluate(test_dataset_combined, [metric_r2, metric_mse])
print(f"Test Pearson R²: {test_scores['pearson_r2_score']:.4f}")
print(f"Test MSE: {test_scores['mean_squared_error']:.4f}")

# Make predictions on the test set
print("\nMaking predictions on test set...")
y_pred = dnn_model.predict(test_dataset_combined)

# Plot predicted vs actual values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r--')
plt.xlabel('Actual Binding Affinity')
plt.ylabel('Predicted Binding Affinity')
plt.title('Predicted vs Actual Binding Affinity')
plt.grid(True)
plt.savefig('output/dti_prediction_scatter.png')
print("Saved prediction scatter plot to output/dti_prediction_scatter.png")

# Save results to a file
results = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred.flatten()
})
results.to_csv('output/dti_prediction_results.csv', index=False)

print("\nResults saved to output/dti_prediction_results.csv")
print("Drug-target interaction prediction completed successfully!") 