"""
Model implementations for drug discovery with DeepChem.
"""

from models.property_prediction import (
    build_random_forest_model,
    build_graph_conv_model,
    build_weave_model,
    build_mpnn_model,
    build_attentive_fp_model,
    build_dnn_model,
    build_transformer_model
)

from models.dti_prediction import (
    build_deepdta_model,
    build_bimodal_gcn_model,
    build_transformer_dti_model,
    build_bilstm_dti_model,
    build_cnn_dti_model
)

from models.generative_models import (
    build_smiles_vae,
    build_junction_tree_vae,
    build_gru_smiles_generator,
    build_conditional_vae,
    build_reinforcement_learning_model,
    calculate_reward
)

__all__ = [
    # Property prediction models
    'build_random_forest_model',
    'build_graph_conv_model',
    'build_weave_model',
    'build_mpnn_model',
    'build_attentive_fp_model',
    'build_dnn_model',
    'build_transformer_model',
    
    # Drug-target interaction models
    'build_deepdta_model',
    'build_bimodal_gcn_model',
    'build_transformer_dti_model',
    'build_bilstm_dti_model',
    'build_cnn_dti_model',
    
    # Generative models
    'build_smiles_vae',
    'build_junction_tree_vae',
    'build_gru_smiles_generator',
    'build_conditional_vae',
    'build_reinforcement_learning_model',
    'calculate_reward'
] 