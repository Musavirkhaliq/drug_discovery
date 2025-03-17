import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Lambda,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding,
    LSTM, GRU, Bidirectional, RepeatVector, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, Crippen
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_smiles_vae(charset_size, max_length=100, latent_dim=196, 
                   intermediate_dim=256, lstm_dim=64, batch_size=32, 
                   learning_rate=0.001, **kwargs):
    """
    Build a Variational Autoencoder (VAE) model for SMILES generation.
    
    Parameters
    ----------
    charset_size : int
        Size of the character set (vocabulary).
    max_length : int, optional (default=100)
        Maximum length of SMILES strings.
    latent_dim : int, optional (default=196)
        Dimension of the latent space.
    intermediate_dim : int, optional (default=256)
        Dimension of the intermediate layer.
    lstm_dim : int, optional (default=64)
        Dimension of the LSTM layer.
    batch_size : int, optional (default=32)
        Batch size for training.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    tuple
        (encoder, decoder, vae) where encoder is the encoder model, decoder is the decoder model,
        and vae is the full VAE model.
    """
    # Encoder
    x = Input(shape=(max_length, charset_size))
    
    # LSTM encoder
    h = LSTM(lstm_dim, return_sequences=True)(x)
    h = LSTM(lstm_dim, return_sequences=False)(h)
    
    # Dense encoder
    h = Dense(intermediate_dim, activation='relu')(h)
    
    # Latent space
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_lstm = LSTM(lstm_dim, return_sequences=True)
    decoder_lstm2 = LSTM(lstm_dim, return_sequences=True)
    decoder_dense = Dense(charset_size, activation='softmax')
    
    # Decoder model
    latent_input = Input(shape=(latent_dim,))
    h_decoded = decoder_h(latent_input)
    h_decoded = RepeatVector(max_length)(h_decoded)
    h_decoded = decoder_lstm(h_decoded)
    h_decoded = decoder_lstm2(h_decoded)
    x_decoded_mean = decoder_dense(h_decoded)
    
    # Full VAE model
    h_decoded = decoder_h(z)
    h_decoded = RepeatVector(max_length)(h_decoded)
    h_decoded = decoder_lstm(h_decoded)
    h_decoded = decoder_lstm2(h_decoded)
    x_decoded_mean = decoder_dense(h_decoded)
    
    # Define models
    vae = Model(x, x_decoded_mean)
    encoder = Model(x, z_mean)
    decoder = Model(latent_input, x_decoded_mean)
    
    # VAE loss
    def vae_loss(x, x_decoded_mean):
        xent_loss = max_length * tf.keras.losses.categorical_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    
    # Compile model
    vae.compile(optimizer=Adam(learning_rate=learning_rate), loss=vae_loss)
    
    logger.info("Built SMILES VAE model")
    logger.info(f"Latent dim: {latent_dim}, Intermediate dim: {intermediate_dim}, LSTM dim: {lstm_dim}")
    
    return encoder, decoder, vae

def build_junction_tree_vae(vocab_size, hidden_size=450, latent_size=56, 
                          depth=3, batch_size=32, learning_rate=0.001, **kwargs):
    """
    Build a Junction Tree Variational Autoencoder (JT-VAE) model for molecule generation.
    
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden_size : int, optional (default=450)
        Size of the hidden layers.
    latent_size : int, optional (default=56)
        Size of the latent space.
    depth : int, optional (default=3)
        Depth of the network.
    batch_size : int, optional (default=32)
        Batch size for training.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    deepchem.models.JTVAEModel
        DeepChem Junction Tree VAE model.
    """
    model = dc.models.JTVAEModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        depth=depth,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )
    
    logger.info("Built Junction Tree VAE model")
    logger.info(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}, Latent size: {latent_size}, Depth: {depth}")
    
    return model

def build_gru_smiles_generator(charset_size, max_length=100, gru_dim=256, 
                             dense_dim=256, latent_dim=64, batch_size=32, 
                             learning_rate=0.001, **kwargs):
    """
    Build a GRU-based SMILES generator model.
    
    Parameters
    ----------
    charset_size : int
        Size of the character set (vocabulary).
    max_length : int, optional (default=100)
        Maximum length of SMILES strings.
    gru_dim : int, optional (default=256)
        Dimension of the GRU layer.
    dense_dim : int, optional (default=256)
        Dimension of the dense layer.
    latent_dim : int, optional (default=64)
        Dimension of the latent space.
    batch_size : int, optional (default=32)
        Batch size for training.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    tuple
        (encoder, decoder, model) where encoder is the encoder model, decoder is the decoder model,
        and model is the full model.
    """
    # Encoder
    encoder_input = Input(shape=(max_length, charset_size))
    encoder_gru = GRU(gru_dim, return_state=True)
    encoder_outputs, state_h = encoder_gru(encoder_input)
    
    # Latent space
    latent = Dense(latent_dim, activation='relu')(state_h)
    
    # Decoder
    decoder_input = Input(shape=(max_length, charset_size))
    decoder_gru = GRU(gru_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_input, initial_state=latent)
    decoder_dense = Dense(charset_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define models
    model = Model([encoder_input, decoder_input], decoder_outputs)
    encoder = Model(encoder_input, latent)
    
    # Decoder model for inference
    decoder_state_input = Input(shape=(latent_dim,))
    decoder_outputs = decoder_gru(decoder_input, initial_state=decoder_state_input)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = Model([decoder_input, decoder_state_input], decoder_outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy'
    )
    
    logger.info("Built GRU SMILES generator model")
    logger.info(f"GRU dim: {gru_dim}, Dense dim: {dense_dim}, Latent dim: {latent_dim}")
    
    return encoder, decoder, model

def build_conditional_vae(charset_size, condition_dim, max_length=100, latent_dim=196, 
                        intermediate_dim=256, lstm_dim=64, batch_size=32, 
                        learning_rate=0.001, **kwargs):
    """
    Build a Conditional Variational Autoencoder (CVAE) model for SMILES generation.
    
    Parameters
    ----------
    charset_size : int
        Size of the character set (vocabulary).
    condition_dim : int
        Dimension of the condition vector.
    max_length : int, optional (default=100)
        Maximum length of SMILES strings.
    latent_dim : int, optional (default=196)
        Dimension of the latent space.
    intermediate_dim : int, optional (default=256)
        Dimension of the intermediate layer.
    lstm_dim : int, optional (default=64)
        Dimension of the LSTM layer.
    batch_size : int, optional (default=32)
        Batch size for training.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    tuple
        (encoder, decoder, cvae) where encoder is the encoder model, decoder is the decoder model,
        and cvae is the full CVAE model.
    """
    # Encoder
    x = Input(shape=(max_length, charset_size))
    c = Input(shape=(condition_dim,))
    
    # Repeat condition for each time step
    c_repeated = RepeatVector(max_length)(c)
    
    # Concatenate input and condition
    h = tf.keras.layers.Concatenate(axis=-1)([x, c_repeated])
    
    # LSTM encoder
    h = LSTM(lstm_dim, return_sequences=True)(h)
    h = LSTM(lstm_dim, return_sequences=False)(h)
    
    # Concatenate LSTM output and condition
    h = tf.keras.layers.Concatenate()([h, c])
    
    # Dense encoder
    h = Dense(intermediate_dim, activation='relu')(h)
    
    # Latent space
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Concatenate latent vector and condition
    z_cond = tf.keras.layers.Concatenate()([z, c])
    
    # Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_lstm = LSTM(lstm_dim, return_sequences=True)
    decoder_lstm2 = LSTM(lstm_dim, return_sequences=True)
    decoder_dense = Dense(charset_size, activation='softmax')
    
    # Decoder model
    latent_input = Input(shape=(latent_dim,))
    c_input = Input(shape=(condition_dim,))
    latent_cond = tf.keras.layers.Concatenate()([latent_input, c_input])
    
    h_decoded = decoder_h(latent_cond)
    h_decoded = RepeatVector(max_length)(h_decoded)
    
    # Repeat condition for each time step
    c_repeated_decoder = RepeatVector(max_length)(c_input)
    h_decoded = tf.keras.layers.Concatenate(axis=-1)([h_decoded, c_repeated_decoder])
    
    h_decoded = decoder_lstm(h_decoded)
    h_decoded = decoder_lstm2(h_decoded)
    x_decoded_mean = decoder_dense(h_decoded)
    
    # Full CVAE model
    h_decoded = decoder_h(z_cond)
    h_decoded = RepeatVector(max_length)(h_decoded)
    
    # Repeat condition for each time step
    c_repeated_full = RepeatVector(max_length)(c)
    h_decoded = tf.keras.layers.Concatenate(axis=-1)([h_decoded, c_repeated_full])
    
    h_decoded = decoder_lstm(h_decoded)
    h_decoded = decoder_lstm2(h_decoded)
    x_decoded_mean = decoder_dense(h_decoded)
    
    # Define models
    cvae = Model([x, c], x_decoded_mean)
    encoder = Model([x, c], z_mean)
    decoder = Model([latent_input, c_input], x_decoded_mean)
    
    # CVAE loss
    def cvae_loss(x, x_decoded_mean):
        xent_loss = max_length * tf.keras.losses.categorical_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    
    # Compile model
    cvae.compile(optimizer=Adam(learning_rate=learning_rate), loss=cvae_loss)
    
    logger.info("Built Conditional SMILES VAE model")
    logger.info(f"Latent dim: {latent_dim}, Intermediate dim: {intermediate_dim}, LSTM dim: {lstm_dim}, Condition dim: {condition_dim}")
    
    return encoder, decoder, cvae

def build_reinforcement_learning_model(charset_size, max_length=100, gru_dim=256, 
                                     dense_dim=256, batch_size=32, 
                                     learning_rate=0.001, **kwargs):
    """
    Build a Reinforcement Learning model for SMILES generation.
    
    Parameters
    ----------
    charset_size : int
        Size of the character set (vocabulary).
    max_length : int, optional (default=100)
        Maximum length of SMILES strings.
    gru_dim : int, optional (default=256)
        Dimension of the GRU layer.
    dense_dim : int, optional (default=256)
        Dimension of the dense layer.
    batch_size : int, optional (default=32)
        Batch size for training.
    learning_rate : float, optional (default=0.001)
        Learning rate for Adam optimizer.
    **kwargs : dict
        Additional arguments for KerasModel.
        
    Returns
    -------
    tuple
        (actor, critic) where actor is the actor model and critic is the critic model.
    """
    # Actor model (policy network)
    actor_input = Input(shape=(max_length, charset_size))
    actor_gru = GRU(gru_dim, return_sequences=True)(actor_input)
    actor_dense = Dense(dense_dim, activation='relu')(actor_gru)
    actor_output = Dense(charset_size, activation='softmax')(actor_dense)
    
    actor = Model(actor_input, actor_output)
    actor.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy'
    )
    
    # Critic model (value network)
    critic_input = Input(shape=(max_length, charset_size))
    critic_gru = GRU(gru_dim)(critic_input)
    critic_dense = Dense(dense_dim, activation='relu')(critic_gru)
    critic_output = Dense(1)(critic_dense)
    
    critic = Model(critic_input, critic_output)
    critic.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    logger.info("Built Reinforcement Learning model for SMILES generation")
    logger.info(f"GRU dim: {gru_dim}, Dense dim: {dense_dim}")
    
    return actor, critic

def calculate_reward(smiles, target_property='qed', target_value=None, maximize=True):
    """
    Calculate reward for a generated SMILES string based on a target property.
    
    Parameters
    ----------
    smiles : str
        SMILES string.
    target_property : str, optional (default='qed')
        Target property to optimize. Options: 'qed', 'logp', 'sa', 'mw'.
    target_value : float, optional
        Target value for the property. If None, maximize or minimize the property.
    maximize : bool, optional (default=True)
        Whether to maximize or minimize the property.
        
    Returns
    -------
    float
        Reward value.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        
        if target_property == 'qed':
            # Drug-likeness score (0-1)
            value = QED.qed(mol)
        elif target_property == 'logp':
            # Lipophilicity
            value = Crippen.MolLogP(mol)
        elif target_property == 'sa':
            # Synthetic accessibility (1-10, lower is better)
            value = -dc.molnet.load_sa_scores([smiles])[1][0]  # Negate for maximization
        elif target_property == 'mw':
            # Molecular weight
            value = Descriptors.MolWt(mol)
        else:
            raise ValueError(f"Target property {target_property} not supported")
        
        if target_value is not None:
            # Reward based on distance to target value
            reward = -abs(value - target_value)
        else:
            # Reward based on maximization or minimization
            reward = value if maximize else -value
        
        return reward
    
    except Exception as e:
        logger.warning(f"Error calculating reward for {smiles}: {e}")
        return 0.0 