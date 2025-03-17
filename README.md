# Drug Discovery with DeepChem

This repository contains a comprehensive drug discovery pipeline using DeepChem and other machine learning tools.

## Project Structure
- `data/`: Contains datasets for training and testing models
- `preprocessing/`: Scripts for data preprocessing and feature engineering
- `models/`: Implementation of various models for drug discovery tasks
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `utils/`: Utility functions for the project

## Features
1. **Property Prediction (QSAR)**
   - Toxicity, solubility, and drug-likeness predictions
   - Models: Random Forest, Graph Convolutional Networks, Transformers

2. **Drug-Target Interaction Prediction (DTI)**
   - Protein sequence + molecular embeddings
   - Models: GNNs, DeepDTA, Transformer-based approaches

3. **Virtual Screening**
   - Prediction of potential drug candidates
   - Ranking based on binding affinity scores

4. **Generative Drug Design**
   - VAEs and reinforcement learning for molecule generation
   - Property optimization

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks in the `notebooks/` directory for examples
3. Use the modular components for your specific drug discovery tasks

## Datasets
The project uses datasets from MoleculeNet, a collection of molecular datasets processed for machine learning.