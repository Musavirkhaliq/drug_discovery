import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_moleculenet_dataset(dataset_name, featurizer='ECFP', split='random', reload=True):
    """
    Load a dataset from MoleculeNet.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. Options include: 'tox21', 'hiv', 'bace_classification',
        'bbbp', 'sider', 'clintox', 'muv', 'pcba', 'toxcast', 'sampl', 'delaney', 'factors',
        'lipo', 'freesolv', 'qm7', 'qm8', 'qm9', 'chembl', 'qm7b', 'qm9'.
    featurizer : str, optional (default='ECFP')
        Type of featurizer to use. Options include: 'ECFP', 'GraphConv', 'Weave', 'Raw', 
        'AdjacencyConv', 'SmilesToImage', 'SmilesToSeq', 'OneHotFeaturizer'.
    split : str, optional (default='random')
        Method of splitting the data. Options include: 'random', 'scaffold', 'stratified', 
        'temporal', 'butina', 'task'.
    reload : bool, optional (default=True)
        Whether to reload the dataset or use cached version.
        
    Returns
    -------
    tuple
        (tasks, datasets, transformers) where tasks is a list of task names, datasets is a 
        tuple of (train, valid, test) datasets, and transformers is a list of transformers 
        applied to the datasets.
    """
    logger.info(f"Loading {dataset_name} dataset with {featurizer} featurizer and {split} split")
    
    # Select featurizer
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = dc.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = dc.feat.RawFeaturizer()
    elif featurizer == 'AdjacencyConv':
        featurizer = dc.feat.AdjacencyFingerprint()
    elif featurizer == 'SmilesToImage':
        featurizer = dc.feat.SmilesToImage()
    elif featurizer == 'SmilesToSeq':
        featurizer = dc.feat.SmilesToSeq()
    elif featurizer == 'OneHotFeaturizer':
        featurizer = dc.feat.OneHotFeaturizer()
    else:
        raise ValueError(f"Featurizer {featurizer} not supported")
    
    # Load dataset
    try:
        tasks, datasets, transformers = dc.molnet.load_dataset(
            dataset_name, featurizer=featurizer, splitter=split, reload=reload
        )
        logger.info(f"Successfully loaded {dataset_name} with {len(tasks)} tasks")
        logger.info(f"Train size: {len(datasets[0].X)}, Valid size: {len(datasets[1].X)}, Test size: {len(datasets[2].X)}")
        return tasks, datasets, transformers
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise

def generate_morgan_fingerprints(smiles_list, radius=2, nBits=2048):
    """
    Generate Morgan fingerprints for a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    radius : int, optional (default=2)
        Radius of Morgan fingerprints.
    nBits : int, optional (default=2048)
        Number of bits in the fingerprint.
        
    Returns
    -------
    numpy.ndarray
        Array of Morgan fingerprints.
    """
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fingerprints.append(np.array(fp))
        else:
            logger.warning(f"Could not convert SMILES to molecule: {smiles}")
            fingerprints.append(np.zeros(nBits))
    return np.array(fingerprints)

def load_custom_dataset(csv_file, smiles_column, target_columns, featurizer='ECFP', split='random', split_ratio=[0.8, 0.1, 0.1]):
    """
    Load a custom dataset from a CSV file.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    smiles_column : str
        Name of the column containing SMILES strings.
    target_columns : list
        List of column names containing target values.
    featurizer : str, optional (default='ECFP')
        Type of featurizer to use.
    split : str, optional (default='random')
        Method of splitting the data.
    split_ratio : list, optional (default=[0.8, 0.1, 0.1])
        Ratio of train, validation, and test sets.
        
    Returns
    -------
    tuple
        (tasks, datasets, transformers) where tasks is a list of task names, datasets is a 
        tuple of (train, valid, test) datasets, and transformers is a list of transformers 
        applied to the datasets.
    """
    logger.info(f"Loading custom dataset from {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Extract SMILES and targets
    smiles_list = df[smiles_column].values
    targets = df[target_columns].values
    
    # Create DeepChem dataset
    if featurizer == 'ECFP':
        features = generate_morgan_fingerprints(smiles_list)
        dataset = dc.data.NumpyDataset(X=features, y=targets, ids=smiles_list)
    else:
        # Use DeepChem featurizers
        if featurizer == 'GraphConv':
            featurizer_obj = dc.feat.ConvMolFeaturizer()
        elif featurizer == 'Weave':
            featurizer_obj = dc.feat.WeaveFeaturizer()
        else:
            raise ValueError(f"Featurizer {featurizer} not supported for custom datasets")
        
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_indices = [i for i, m in enumerate(mols) if m is not None]
        valid_mols = [mols[i] for i in valid_indices]
        valid_targets = targets[valid_indices]
        
        features = featurizer_obj.featurize(valid_mols)
        dataset = dc.data.NumpyDataset(X=features, y=valid_targets, ids=[smiles_list[i] for i in valid_indices])
    
    # Split dataset
    if split == 'random':
        splitter = dc.splits.RandomSplitter()
    elif split == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    elif split == 'stratified':
        splitter = dc.splits.RandomStratifiedSplitter()
    else:
        raise ValueError(f"Split method {split} not supported for custom datasets")
    
    train, valid, test = splitter.train_valid_test_split(
        dataset, frac_train=split_ratio[0], frac_valid=split_ratio[1], frac_test=split_ratio[2]
    )
    
    logger.info(f"Successfully loaded custom dataset with {len(target_columns)} tasks")
    logger.info(f"Train size: {len(train.X)}, Valid size: {len(valid.X)}, Test size: {len(test.X)}")
    
    return target_columns, (train, valid, test), [] 