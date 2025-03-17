import numpy as np
import deepchem as dc
from deepchem.models import GraphConvModel, WeaveModel, ScScoreModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_random_forest_model(mode='regression', n_estimators=100, max_depth=None, **kwargs):
    """
    Build a Random Forest model for property prediction.
    
    Parameters
    ----------
    mode : str, optional (default='regression')
        Mode of the model. Options: 'regression', 'classification'.
    n_estimators : int, optional (default=100)
        Number of trees in the forest.
    max_depth : int, optional (default=None)
        Maximum depth of the trees.
    **kwargs : dict
        Additional arguments for RandomForestRegressor or RandomForestClassifier.
        
    Returns
    -------
    deepchem.models.SklearnModel
        DeepChem wrapper for scikit-learn Random Forest model.
    """
    if mode == 'regression':
        sklearn_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            **kwargs
        )
        model = dc.models.SklearnModel(sklearn_model)
        logger.info(f"Built Random Forest regression model with {n_estimators} estimators")
    elif mode == 'classification':
        sklearn_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            **kwargs
        )
        model = dc.models.SklearnModel(sklearn_model)
        logger.info(f"Built Random Forest classification model with {n_estimators} estimators")
    else:
        raise ValueError(f"Mode {mode} not supported")
    
    return model

def build_graph_conv_model(n_tasks, mode='regression', graph_conv_layers=[64, 64], 
                          dense_layers=[128], dropout=0.2, learning_rate=0.001, **kwargs):
    """
    Build a Graph Convolutional Network model for property prediction.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks (output dimensions).
    mode : str, optional (default='regression')
        Mode of the model. Options: 'regression', 'classification'.
    graph_conv_layers : list, optional (default=[64, 64])
        List of hidden units in graph convolutional layers.
    dense_layers : list, optional (default=[128])
        List of hidden units in dense layers.
    dropout : float, optional (default=0.2)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for GraphConvModel.
        
    Returns
    -------
    deepchem.models.GraphConvModel
        DeepChem Graph Convolutional Network model.
    """
    model = GraphConvModel(
        n_tasks=n_tasks,
        mode=mode,
        graph_conv_layers=graph_conv_layers,
        dense_layer_size=dense_layers[0] if dense_layers else 128,
        dropout=dropout,
        learning_rate=learning_rate,
        **kwargs
    )
    
    logger.info(f"Built Graph Convolutional Network model for {mode} with {n_tasks} tasks")
    logger.info(f"Graph conv layers: {graph_conv_layers}, Dense layers: {dense_layers}, Dropout: {dropout}")
    
    return model

def build_weave_model(n_tasks, mode='regression', n_graph_feat=128, n_pair_feat=14,
                     n_hidden=50, n_depth=3, learning_rate=0.001, **kwargs):
    """
    Build a Weave model for property prediction.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks (output dimensions).
    mode : str, optional (default='regression')
        Mode of the model. Options: 'regression', 'classification'.
    n_graph_feat : int, optional (default=128)
        Number of graph features.
    n_pair_feat : int, optional (default=14)
        Number of pair features.
    n_hidden : int, optional (default=50)
        Number of hidden units in the network.
    n_depth : int, optional (default=3)
        Number of layers in the network.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for WeaveModel.
        
    Returns
    -------
    deepchem.models.WeaveModel
        DeepChem Weave model.
    """
    model = WeaveModel(
        n_tasks=n_tasks,
        mode=mode,
        n_graph_feat=n_graph_feat,
        n_pair_feat=n_pair_feat,
        n_hidden=n_hidden,
        n_depth=n_depth,
        learning_rate=learning_rate,
        **kwargs
    )
    
    logger.info(f"Built Weave model for {mode} with {n_tasks} tasks")
    logger.info(f"Graph features: {n_graph_feat}, Pair features: {n_pair_feat}, Hidden units: {n_hidden}, Depth: {n_depth}")
    
    return model

def build_mpnn_model(n_tasks, mode='regression', n_atom_feat=75, n_pair_feat=14,
                    n_hidden=100, T=5, M=10, learning_rate=0.001, **kwargs):
    """
    Build a Message Passing Neural Network model for property prediction.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks (output dimensions).
    mode : str, optional (default='regression')
        Mode of the model. Options: 'regression', 'classification'.
    n_atom_feat : int, optional (default=75)
        Number of atom features.
    n_pair_feat : int, optional (default=14)
        Number of pair features.
    n_hidden : int, optional (default=100)
        Number of hidden units in the network.
    T : int, optional (default=5)
        Number of message passing steps.
    M : int, optional (default=10)
        Number of hidden units in the readout function.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for MPNNModel.
        
    Returns
    -------
    deepchem.models.MPNNModel
        DeepChem Message Passing Neural Network model.
    """
    model = dc.models.MPNNModel(
        n_tasks=n_tasks,
        mode=mode,
        n_atom_feat=n_atom_feat,
        n_pair_feat=n_pair_feat,
        n_hidden=n_hidden,
        T=T,
        M=M,
        learning_rate=learning_rate,
        **kwargs
    )
    
    logger.info(f"Built Message Passing Neural Network model for {mode} with {n_tasks} tasks")
    logger.info(f"Atom features: {n_atom_feat}, Pair features: {n_pair_feat}, Hidden units: {n_hidden}, T: {T}, M: {M}")
    
    return model

def build_attentive_fp_model(n_tasks, mode='regression', num_layers=3, num_timesteps=3,
                           graph_feat_size=200, dropout=0.2, learning_rate=0.001, **kwargs):
    """
    Build an Attentive Fingerprint model for property prediction.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks (output dimensions).
    mode : str, optional (default='regression')
        Mode of the model. Options: 'regression', 'classification'.
    num_layers : int, optional (default=3)
        Number of layers in the network.
    num_timesteps : int, optional (default=3)
        Number of timesteps for message passing.
    graph_feat_size : int, optional (default=200)
        Size of graph features.
    dropout : float, optional (default=0.2)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for AttentiveFPModel.
        
    Returns
    -------
    deepchem.models.AttentiveFPModel
        DeepChem Attentive Fingerprint model.
    """
    model = dc.models.AttentiveFPModel(
        n_tasks=n_tasks,
        mode=mode,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        graph_feat_size=graph_feat_size,
        dropout=dropout,
        learning_rate=learning_rate,
        **kwargs
    )
    
    logger.info(f"Built Attentive Fingerprint model for {mode} with {n_tasks} tasks")
    logger.info(f"Layers: {num_layers}, Timesteps: {num_timesteps}, Graph feat size: {graph_feat_size}, Dropout: {dropout}")
    
    return model

def build_dnn_model(n_tasks, input_dim, mode='regression', hidden_layers=[1024, 512, 128],
                  activation='relu', dropout=0.2, learning_rate=0.001, batch_size=32, **kwargs):
    """
    Build a Deep Neural Network model for property prediction.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks (output dimensions).
    input_dim : int
        Input dimension (number of features).
    mode : str, optional (default='regression')
        Mode of the model. Options: 'regression', 'classification'.
    hidden_layers : list, optional (default=[1024, 512, 128])
        List of hidden units in dense layers.
    activation : str, optional (default='relu')
        Activation function.
    dropout : float, optional (default=0.2)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    batch_size : int, optional (default=32)
        Batch size for training.
    **kwargs : dict
        Additional arguments for MultitaskClassifier or MultitaskRegressor.
        
    Returns
    -------
    deepchem.models.MultitaskClassifier or deepchem.models.MultitaskRegressor
        DeepChem Deep Neural Network model.
    """
    # Define model architecture
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    # Add hidden layers
    for units in hidden_layers:
        x = Dense(units, activation=activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    
    # Add output layer
    if mode == 'regression':
        outputs = Dense(n_tasks, activation='linear')(x)
    elif mode == 'classification':
        outputs = Dense(n_tasks, activation='sigmoid')(x)
    else:
        raise ValueError(f"Mode {mode} not supported")
    
    # Create model
    keras_model = Model(inputs=inputs, outputs=outputs)
    keras_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse' if mode == 'regression' else 'binary_crossentropy',
        metrics=['mae'] if mode == 'regression' else ['accuracy']
    )
    
    # Wrap in DeepChem model
    if mode == 'regression':
        model = dc.models.KerasModel(
            keras_model,
            loss='mse',
            output_types=['prediction'],
            batch_size=batch_size,
            **kwargs
        )
    elif mode == 'classification':
        model = dc.models.KerasModel(
            keras_model,
            loss='binary_crossentropy',
            output_types=['prediction'],
            batch_size=batch_size,
            **kwargs
        )
    
    logger.info(f"Built Deep Neural Network model for {mode} with {n_tasks} tasks")
    logger.info(f"Input dim: {input_dim}, Hidden layers: {hidden_layers}, Activation: {activation}, Dropout: {dropout}")
    
    return model

def build_transformer_model(n_tasks, input_dim, mode='regression', num_layers=2, num_attention_heads=2,
                          hidden_dim=128, dropout=0.1, learning_rate=0.001, batch_size=32, **kwargs):
    """
    Build a Transformer model for property prediction.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks (output dimensions).
    input_dim : int
        Input dimension (number of features).
    mode : str, optional (default='regression')
        Mode of the model. Options: 'regression', 'classification'.
    num_layers : int, optional (default=2)
        Number of transformer layers.
    num_attention_heads : int, optional (default=2)
        Number of attention heads.
    hidden_dim : int, optional (default=128)
        Hidden dimension.
    dropout : float, optional (default=0.1)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    batch_size : int, optional (default=32)
        Batch size for training.
    **kwargs : dict
        Additional arguments for MultitaskClassifier or MultitaskRegressor.
        
    Returns
    -------
    deepchem.models.MultitaskClassifier or deepchem.models.MultitaskRegressor
        DeepChem Transformer model.
    """
    # Define model architecture
    inputs = Input(shape=(input_dim,))
    x = Dense(hidden_dim)(inputs)  # Project to hidden dimension
    
    # Add transformer layers
    for _ in range(num_layers):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_dim // num_attention_heads
        )(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = Dense(hidden_dim * 4, activation='relu')(x)
        ffn = Dense(hidden_dim)(ffn)
        x = tf.keras.layers.Add()([x, ffn])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(tf.expand_dims(x, axis=1))
    
    # Add output layer
    if mode == 'regression':
        outputs = Dense(n_tasks, activation='linear')(x)
    elif mode == 'classification':
        outputs = Dense(n_tasks, activation='sigmoid')(x)
    else:
        raise ValueError(f"Mode {mode} not supported")
    
    # Create model
    keras_model = Model(inputs=inputs, outputs=outputs)
    keras_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse' if mode == 'regression' else 'binary_crossentropy',
        metrics=['mae'] if mode == 'regression' else ['accuracy']
    )
    
    # Wrap in DeepChem model
    if mode == 'regression':
        model = dc.models.KerasModel(
            keras_model,
            loss='mse',
            output_types=['prediction'],
            batch_size=batch_size,
            **kwargs
        )
    elif mode == 'classification':
        model = dc.models.KerasModel(
            keras_model,
            loss='binary_crossentropy',
            output_types=['prediction'],
            batch_size=batch_size,
            **kwargs
        )
    
    logger.info(f"Built Transformer model for {mode} with {n_tasks} tasks")
    logger.info(f"Input dim: {input_dim}, Layers: {num_layers}, Attention heads: {num_attention_heads}, Hidden dim: {hidden_dim}")
    
    return model 