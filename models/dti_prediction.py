import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Concatenate,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding,
    LSTM, Bidirectional, Attention, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
import deepchem as dc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_deepdta_model(protein_input_dim, drug_input_dim, protein_length=1000, drug_length=100,
                       filters=[32, 64, 96], kernel_sizes=[4, 8, 12], dense_layers=[1024, 512, 256],
                       dropout=0.2, learning_rate=0.001, batch_size=32, **kwargs):
    """
    Build a DeepDTA model for drug-target interaction prediction.
    
    Parameters
    ----------
    protein_input_dim : int
        Dimension of protein input features.
    drug_input_dim : int
        Dimension of drug input features.
    protein_length : int, optional (default=1000)
        Length of protein sequence.
    drug_length : int, optional (default=100)
        Length of drug representation.
    filters : list, optional (default=[32, 64, 96])
        List of filter sizes for convolutional layers.
    kernel_sizes : list, optional (default=[4, 8, 12])
        List of kernel sizes for convolutional layers.
    dense_layers : list, optional (default=[1024, 512, 256])
        List of hidden units in dense layers.
    dropout : float, optional (default=0.2)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    batch_size : int, optional (default=32)
        Batch size for training.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    deepchem.models.KerasModel
        DeepChem wrapper for DeepDTA model.
    """
    # Protein input branch
    protein_input = Input(shape=(protein_length, protein_input_dim))
    p = protein_input
    
    # Convolutional layers for protein
    for i in range(len(filters)):
        p = Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], activation='relu', padding='same')(p)
        p = MaxPooling1D(pool_size=2)(p)
    
    p = GlobalAveragePooling1D()(p)
    
    # Drug input branch
    drug_input = Input(shape=(drug_length, drug_input_dim))
    d = drug_input
    
    # Convolutional layers for drug
    for i in range(len(filters)):
        d = Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], activation='relu', padding='same')(d)
        d = MaxPooling1D(pool_size=2)(d)
    
    d = GlobalAveragePooling1D()(d)
    
    # Concatenate protein and drug features
    x = Concatenate()([p, d])
    
    # Dense layers
    for units in dense_layers:
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=[protein_input, drug_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Wrap in DeepChem model
    dc_model = dc.models.KerasModel(
        model,
        loss='binary_crossentropy',
        output_types=['prediction'],
        batch_size=batch_size,
        **kwargs
    )
    
    logger.info("Built DeepDTA model for drug-target interaction prediction")
    logger.info(f"Filters: {filters}, Kernel sizes: {kernel_sizes}, Dense layers: {dense_layers}, Dropout: {dropout}")
    
    return dc_model

def build_bimodal_gcn_model(n_tasks=1, graph_conv_layers=[64, 64], dense_layers=[128],
                          protein_embedding_dim=128, protein_filter_sizes=[4, 8, 12],
                          protein_num_filters=[32, 64, 96], dropout=0.2,
                          learning_rate=0.001, **kwargs):
    """
    Build a bimodal Graph Convolutional Network model for drug-target interaction prediction.
    
    Parameters
    ----------
    n_tasks : int, optional (default=1)
        Number of tasks (output dimensions).
    graph_conv_layers : list, optional (default=[64, 64])
        List of hidden units in graph convolutional layers.
    dense_layers : list, optional (default=[128])
        List of hidden units in dense layers.
    protein_embedding_dim : int, optional (default=128)
        Dimension of protein embedding.
    protein_filter_sizes : list, optional (default=[4, 8, 12])
        List of filter sizes for protein convolutional layers.
    protein_num_filters : list, optional (default=[32, 64, 96])
        List of number of filters for protein convolutional layers.
    dropout : float, optional (default=0.2)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for GraphConvModel.
        
    Returns
    -------
    deepchem.models.GraphConvModel
        DeepChem Graph Convolutional Network model for drug-target interaction.
    """
    model = dc.models.DTIGraphConvModel(
        n_tasks=n_tasks,
        graph_conv_layers=graph_conv_layers,
        dense_layer_size=dense_layers[0] if dense_layers else 128,
        dropout=dropout,
        learning_rate=learning_rate,
        protein_embedding_dim=protein_embedding_dim,
        protein_filter_sizes=protein_filter_sizes,
        protein_num_filters=protein_num_filters,
        **kwargs
    )
    
    logger.info("Built bimodal Graph Convolutional Network model for drug-target interaction prediction")
    logger.info(f"Graph conv layers: {graph_conv_layers}, Dense layers: {dense_layers}, Dropout: {dropout}")
    logger.info(f"Protein embedding dim: {protein_embedding_dim}, Protein filters: {protein_num_filters}")
    
    return model

def build_transformer_dti_model(protein_vocab_size, drug_vocab_size, protein_length=1000, drug_length=100,
                              embedding_dim=128, num_heads=8, ff_dim=128, num_transformer_blocks=2,
                              dropout=0.1, learning_rate=0.001, batch_size=32, **kwargs):
    """
    Build a Transformer model for drug-target interaction prediction.
    
    Parameters
    ----------
    protein_vocab_size : int
        Size of protein vocabulary.
    drug_vocab_size : int
        Size of drug vocabulary.
    protein_length : int, optional (default=1000)
        Length of protein sequence.
    drug_length : int, optional (default=100)
        Length of drug representation.
    embedding_dim : int, optional (default=128)
        Dimension of embedding.
    num_heads : int, optional (default=8)
        Number of attention heads.
    ff_dim : int, optional (default=128)
        Hidden dimension in feed-forward network.
    num_transformer_blocks : int, optional (default=2)
        Number of transformer blocks.
    dropout : float, optional (default=0.1)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    batch_size : int, optional (default=32)
        Batch size for training.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    deepchem.models.KerasModel
        DeepChem wrapper for Transformer model.
    """
    # Protein input branch
    protein_input = Input(shape=(protein_length,))
    protein_embedding = Embedding(protein_vocab_size, embedding_dim)(protein_input)
    
    # Transformer blocks for protein
    p = protein_embedding
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads
        )(p, p)
        p = tf.keras.layers.Add()([p, attention_output])
        p = tf.keras.layers.LayerNormalization(epsilon=1e-6)(p)
        
        # Feed-forward network
        ffn = Dense(ff_dim, activation='relu')(p)
        ffn = Dense(embedding_dim)(ffn)
        p = tf.keras.layers.Add()([p, ffn])
        p = tf.keras.layers.LayerNormalization(epsilon=1e-6)(p)
        p = Dropout(dropout)(p)
    
    # Global average pooling for protein
    p = GlobalAveragePooling1D()(p)
    
    # Drug input branch
    drug_input = Input(shape=(drug_length,))
    drug_embedding = Embedding(drug_vocab_size, embedding_dim)(drug_input)
    
    # Transformer blocks for drug
    d = drug_embedding
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads
        )(d, d)
        d = tf.keras.layers.Add()([d, attention_output])
        d = tf.keras.layers.LayerNormalization(epsilon=1e-6)(d)
        
        # Feed-forward network
        ffn = Dense(ff_dim, activation='relu')(d)
        ffn = Dense(embedding_dim)(ffn)
        d = tf.keras.layers.Add()([d, ffn])
        d = tf.keras.layers.LayerNormalization(epsilon=1e-6)(d)
        d = Dropout(dropout)(d)
    
    # Global average pooling for drug
    d = GlobalAveragePooling1D()(d)
    
    # Concatenate protein and drug features
    x = Concatenate()([p, d])
    
    # Dense layers
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim // 2, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=[protein_input, drug_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Wrap in DeepChem model
    dc_model = dc.models.KerasModel(
        model,
        loss='binary_crossentropy',
        output_types=['prediction'],
        batch_size=batch_size,
        **kwargs
    )
    
    logger.info("Built Transformer model for drug-target interaction prediction")
    logger.info(f"Embedding dim: {embedding_dim}, Num heads: {num_heads}, FF dim: {ff_dim}, Transformer blocks: {num_transformer_blocks}")
    
    return dc_model

def build_bilstm_dti_model(protein_vocab_size, drug_input_dim, protein_length=1000, drug_length=100,
                         embedding_dim=128, lstm_units=64, dense_layers=[256, 128],
                         dropout=0.2, learning_rate=0.001, batch_size=32, **kwargs):
    """
    Build a BiLSTM model for drug-target interaction prediction.
    
    Parameters
    ----------
    protein_vocab_size : int
        Size of protein vocabulary.
    drug_input_dim : int
        Dimension of drug input features.
    protein_length : int, optional (default=1000)
        Length of protein sequence.
    drug_length : int, optional (default=100)
        Length of drug representation.
    embedding_dim : int, optional (default=128)
        Dimension of protein embedding.
    lstm_units : int, optional (default=64)
        Number of LSTM units.
    dense_layers : list, optional (default=[256, 128])
        List of hidden units in dense layers.
    dropout : float, optional (default=0.2)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    batch_size : int, optional (default=32)
        Batch size for training.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    deepchem.models.KerasModel
        DeepChem wrapper for BiLSTM model.
    """
    # Protein input branch
    protein_input = Input(shape=(protein_length,))
    protein_embedding = Embedding(protein_vocab_size, embedding_dim)(protein_input)
    p = Bidirectional(LSTM(lstm_units, return_sequences=True))(protein_embedding)
    p = GlobalAveragePooling1D()(p)
    
    # Drug input branch
    drug_input = Input(shape=(drug_length, drug_input_dim))
    d = Bidirectional(LSTM(lstm_units, return_sequences=True))(drug_input)
    d = GlobalAveragePooling1D()(d)
    
    # Concatenate protein and drug features
    x = Concatenate()([p, d])
    
    # Dense layers
    for units in dense_layers:
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=[protein_input, drug_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Wrap in DeepChem model
    dc_model = dc.models.KerasModel(
        model,
        loss='binary_crossentropy',
        output_types=['prediction'],
        batch_size=batch_size,
        **kwargs
    )
    
    logger.info("Built BiLSTM model for drug-target interaction prediction")
    logger.info(f"Embedding dim: {embedding_dim}, LSTM units: {lstm_units}, Dense layers: {dense_layers}, Dropout: {dropout}")
    
    return dc_model

def build_cnn_dti_model(protein_vocab_size, drug_input_dim, protein_length=1000, drug_length=100,
                      embedding_dim=128, filters=[32, 64, 128], kernel_sizes=[3, 5, 7],
                      dense_layers=[256, 128], dropout=0.2, learning_rate=0.001, batch_size=32, **kwargs):
    """
    Build a CNN model for drug-target interaction prediction.
    
    Parameters
    ----------
    protein_vocab_size : int
        Size of protein vocabulary.
    drug_input_dim : int
        Dimension of drug input features.
    protein_length : int, optional (default=1000)
        Length of protein sequence.
    drug_length : int, optional (default=100)
        Length of drug representation.
    embedding_dim : int, optional (default=128)
        Dimension of protein embedding.
    filters : list, optional (default=[32, 64, 128])
        List of filter sizes for convolutional layers.
    kernel_sizes : list, optional (default=[3, 5, 7])
        List of kernel sizes for convolutional layers.
    dense_layers : list, optional (default=[256, 128])
        List of hidden units in dense layers.
    dropout : float, optional (default=0.2)
        Dropout rate.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    batch_size : int, optional (default=32)
        Batch size for training.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    deepchem.models.KerasModel
        DeepChem wrapper for CNN model.
    """
    # Protein input branch
    protein_input = Input(shape=(protein_length,))
    protein_embedding = Embedding(protein_vocab_size, embedding_dim)(protein_input)
    
    # Parallel convolutional layers for protein
    p_convs = []
    for i in range(len(filters)):
        conv = Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], activation='relu', padding='same')(protein_embedding)
        pool = MaxPooling1D(pool_size=2)(conv)
        p_convs.append(pool)
    
    # Concatenate protein convolutional outputs
    if len(p_convs) > 1:
        p = Concatenate()(p_convs)
    else:
        p = p_convs[0]
    
    p = GlobalAveragePooling1D()(p)
    
    # Drug input branch
    drug_input = Input(shape=(drug_length, drug_input_dim))
    
    # Parallel convolutional layers for drug
    d_convs = []
    for i in range(len(filters)):
        conv = Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], activation='relu', padding='same')(drug_input)
        pool = MaxPooling1D(pool_size=2)(conv)
        d_convs.append(pool)
    
    # Concatenate drug convolutional outputs
    if len(d_convs) > 1:
        d = Concatenate()(d_convs)
    else:
        d = d_convs[0]
    
    d = GlobalAveragePooling1D()(d)
    
    # Concatenate protein and drug features
    x = Concatenate()([p, d])
    
    # Dense layers
    for units in dense_layers:
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=[protein_input, drug_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Wrap in DeepChem model
    dc_model = dc.models.KerasModel(
        model,
        loss='binary_crossentropy',
        output_types=['prediction'],
        batch_size=batch_size,
        **kwargs
    )
    
    logger.info("Built CNN model for drug-target interaction prediction")
    logger.info(f"Embedding dim: {embedding_dim}, Filters: {filters}, Kernel sizes: {kernel_sizes}, Dense layers: {dense_layers}")
    
    return dc_model 