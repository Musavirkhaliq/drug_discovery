import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, MACCSkeys
from rdkit.ML.Descriptors import MolecularDescriptriptorCalculator
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_fingerprints(smiles_list, fp_type='morgan', radius=2, nBits=2048):
    """
    Generate molecular fingerprints for a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    fp_type : str, optional (default='morgan')
        Type of fingerprint to generate. Options: 'morgan', 'maccs', 'rdkit', 'atom_pair'.
    radius : int, optional (default=2)
        Radius for Morgan fingerprints.
    nBits : int, optional (default=2048)
        Number of bits in the fingerprint.
        
    Returns
    -------
    numpy.ndarray
        Array of fingerprints.
    """
    fingerprints = []
    valid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not convert SMILES to molecule: {smiles}")
            continue
        
        if fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        elif fp_type == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, fpSize=nBits)
        elif fp_type == 'atom_pair':
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
        else:
            raise ValueError(f"Fingerprint type {fp_type} not supported")
        
        fingerprints.append(np.array(fp))
        valid_smiles.append(smiles)
    
    logger.info(f"Generated {len(fingerprints)} {fp_type} fingerprints")
    
    return np.array(fingerprints), valid_smiles

def calculate_descriptors(smiles_list, descriptor_list=None):
    """
    Calculate molecular descriptors for a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    descriptor_list : list, optional
        List of descriptor names to calculate. If None, calculate all available descriptors.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame of molecular descriptors.
    """
    # Get all available descriptors if not specified
    if descriptor_list is None:
        descriptor_list = [x[0] for x in Descriptors._descList]
    
    # Initialize calculator
    calculator = MolecularDescriptriptorCalculator.MolecularDescriptorCalculator(descriptor_list)
    
    # Calculate descriptors
    descriptors = []
    valid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not convert SMILES to molecule: {smiles}")
            continue
        
        # Calculate descriptors
        try:
            desc_values = calculator.CalcDescriptors(mol)
            descriptors.append(desc_values)
            valid_smiles.append(smiles)
        except Exception as e:
            logger.warning(f"Error calculating descriptors for {smiles}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(descriptors, columns=descriptor_list)
    df['SMILES'] = valid_smiles
    
    logger.info(f"Calculated {len(descriptor_list)} descriptors for {len(descriptors)} molecules")
    
    return df

def generate_graph_features(smiles_list, use_edges=True, use_chirality=True):
    """
    Generate graph features for a list of SMILES strings using DeepChem's MolGraphConvFeaturizer.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    use_edges : bool, optional (default=True)
        Whether to use edge features.
    use_chirality : bool, optional (default=True)
        Whether to use chirality information.
        
    Returns
    -------
    list
        List of graph features.
    """
    featurizer = MolGraphConvFeaturizer(use_edges=use_edges, use_chirality=use_chirality)
    
    # Generate features
    features = []
    valid_smiles = []
    
    for smiles in smiles_list:
        try:
            feat = featurizer.featurize(smiles)
            if feat and len(feat) > 0:
                features.append(feat[0])
                valid_smiles.append(smiles)
            else:
                logger.warning(f"Failed to generate graph features for {smiles}")
        except Exception as e:
            logger.warning(f"Error generating graph features for {smiles}: {e}")
    
    logger.info(f"Generated graph features for {len(features)} molecules")
    
    return features, valid_smiles

def generate_lipinski_features(smiles_list):
    """
    Calculate Lipinski's Rule of Five features for a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame of Lipinski features.
    """
    lipinski_features = []
    valid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not convert SMILES to molecule: {smiles}")
            continue
        
        features = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': Descriptors.TPSA(mol),
            'Lipinski_Violations': Lipinski.NumRotatableBonds(mol)
        }
        
        # Calculate Lipinski violations
        violations = 0
        if features['MW'] > 500: violations += 1
        if features['LogP'] > 5: violations += 1
        if features['HBD'] > 5: violations += 1
        if features['HBA'] > 10: violations += 1
        
        features['Lipinski_Violations'] = violations
        
        lipinski_features.append(features)
        valid_smiles.append(smiles)
    
    # Create DataFrame
    df = pd.DataFrame(lipinski_features)
    df['SMILES'] = valid_smiles
    
    logger.info(f"Calculated Lipinski features for {len(lipinski_features)} molecules")
    
    return df

def normalize_features(features, scaler_type='standard'):
    """
    Normalize features using different scaling methods.
    
    Parameters
    ----------
    features : numpy.ndarray
        Array of features.
    scaler_type : str, optional (default='standard')
        Type of scaler to use. Options: 'standard', 'minmax', 'robust'.
        
    Returns
    -------
    tuple
        (normalized_features, scaler)
    """
    if scaler_type == 'standard':
        scaler = dc.trans.StandardScaler()
    elif scaler_type == 'minmax':
        scaler = dc.trans.MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = dc.trans.RobustScaler()
    else:
        raise ValueError(f"Scaler type {scaler_type} not supported")
    
    normalized_features = scaler.transform(features)
    
    logger.info(f"Normalized features using {scaler_type} scaling")
    
    return normalized_features, scaler

def augment_data(smiles_list, augmentation_factor=5):
    """
    Augment data by generating SMILES variants.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    augmentation_factor : int, optional (default=5)
        Number of augmented SMILES to generate per original SMILES.
        
    Returns
    -------
    list
        List of augmented SMILES strings.
    """
    augmented_smiles = []
    original_to_augmented = {}
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not convert SMILES to molecule: {smiles}")
            continue
        
        # Add original SMILES
        augmented_smiles.append(smiles)
        original_to_augmented[smiles] = [smiles]
        
        # Generate random SMILES
        for _ in range(augmentation_factor - 1):
            try:
                # Randomize atom order
                random_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, doRandom=True))
                if random_mol is not None:
                    random_smiles = Chem.MolToSmiles(random_mol)
                    augmented_smiles.append(random_smiles)
                    original_to_augmented[smiles].append(random_smiles)
            except Exception as e:
                logger.warning(f"Error generating augmented SMILES for {smiles}: {e}")
    
    logger.info(f"Generated {len(augmented_smiles)} augmented SMILES from {len(smiles_list)} original SMILES")
    
    return augmented_smiles, original_to_augmented

def featurize_protein_sequence(sequences, method='amino_acid_composition'):
    """
    Featurize protein sequences.
    
    Parameters
    ----------
    sequences : list
        List of protein sequences.
    method : str, optional (default='amino_acid_composition')
        Method for featurizing protein sequences. Options: 'amino_acid_composition', 'sequence_conv'.
        
    Returns
    -------
    numpy.ndarray
        Array of protein features.
    """
    if method == 'amino_acid_composition':
        # Calculate amino acid composition
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        features = []
        
        for seq in sequences:
            # Count amino acids
            counts = {aa: seq.count(aa) / len(seq) for aa in amino_acids}
            features.append([counts[aa] for aa in amino_acids])
        
        return np.array(features)
    
    elif method == 'sequence_conv':
        # Use DeepChem's SequenceConvFeaturizer
        featurizer = dc.feat.SequenceConvFeaturizer()
        return featurizer.featurize(sequences)
    
    else:
        raise ValueError(f"Method {method} not supported for protein featurization")

def combine_molecule_protein_features(mol_features, protein_features):
    """
    Combine molecule and protein features for drug-target interaction prediction.
    
    Parameters
    ----------
    mol_features : numpy.ndarray
        Array of molecule features.
    protein_features : numpy.ndarray
        Array of protein features.
        
    Returns
    -------
    numpy.ndarray
        Array of combined features.
    """
    # Simple concatenation
    combined_features = np.concatenate([mol_features, protein_features], axis=1)
    
    logger.info(f"Combined molecule features (shape: {mol_features.shape}) and protein features (shape: {protein_features.shape})")
    
    return combined_features 