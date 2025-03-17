import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Lipinski
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import io
from PIL import Image
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def smiles_to_mol(smiles):
    """
    Convert SMILES string to RDKit molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES string.
        
    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not convert SMILES to molecule: {smiles}")
    return mol

def visualize_molecule(smiles, molSize=(400, 300), kekulize=True, title=None):
    """
    Visualize a molecule from SMILES string.
    
    Parameters
    ----------
    smiles : str
        SMILES string.
    molSize : tuple, optional (default=(400, 300))
        Size of the molecule image.
    kekulize : bool, optional (default=True)
        Whether to kekulize the molecule.
    title : str, optional
        Title of the plot.
        
    Returns
    -------
    PIL.Image
        Image of the molecule.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    # Generate 2D coordinates if not available
    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)
    
    # Draw molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert to PIL image
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    
    # Display with title if provided
    if title:
        plt.figure(figsize=(molSize[0]/100, molSize[1]/100))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    return img

def visualize_molecules(smiles_list, mols_per_row=4, molSize=(200, 150), titles=None, legends=None):
    """
    Visualize multiple molecules from SMILES strings.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    mols_per_row : int, optional (default=4)
        Number of molecules per row.
    molSize : tuple, optional (default=(200, 150))
        Size of each molecule image.
    titles : list, optional
        List of titles for each molecule.
    legends : list, optional
        List of legends for each molecule.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with molecule images.
    """
    mols = [smiles_to_mol(smiles) for smiles in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    
    if not mols:
        logger.warning("No valid molecules to visualize")
        return None
    
    if titles is None:
        titles = [None] * len(mols)
    
    # Draw molecules
    img = Draw.MolsToGridImage(
        mols, 
        molsPerRow=mols_per_row, 
        subImgSize=molSize, 
        legends=legends if legends else None
    )
    
    # Display
    plt.figure(figsize=(molSize[0]*mols_per_row/100, molSize[1]*((len(mols)-1)//mols_per_row + 1)/100))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img

def calculate_molecular_properties(smiles):
    """
    Calculate molecular properties for a SMILES string.
    
    Parameters
    ----------
    smiles : str
        SMILES string.
        
    Returns
    -------
    dict
        Dictionary of molecular properties.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    properties = {
        'Molecular Weight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBond Donors': Lipinski.NumHDonors(mol),
        'HBond Acceptors': Lipinski.NumHAcceptors(mol),
        'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Num Rings': Descriptors.RingCount(mol),
        'Num Aromatic Rings': Descriptors.NumAromaticRings(mol),
        'Num Atoms': mol.GetNumAtoms(),
        'Num Heavy Atoms': mol.GetNumHeavyAtoms(),
        'Num Bonds': mol.GetNumBonds()
    }
    
    return properties

def plot_property_distribution(properties_list, property_name, bins=20, save_path=None):
    """
    Plot distribution of a molecular property.
    
    Parameters
    ----------
    properties_list : list
        List of property dictionaries.
    property_name : str
        Name of the property to plot.
    bins : int, optional (default=20)
        Number of bins for histogram.
    save_path : str, optional
        Path to save the plot.
    """
    values = [props[property_name] for props in properties_list if props is not None and property_name in props]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=bins, kde=True)
    plt.title(f'Distribution of {property_name}')
    plt.xlabel(property_name)
    plt.ylabel('Frequency')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_property_correlation(properties_list, property_names=None, save_path=None):
    """
    Plot correlation matrix of molecular properties.
    
    Parameters
    ----------
    properties_list : list
        List of property dictionaries.
    property_names : list, optional
        List of property names to include in the correlation matrix.
    save_path : str, optional
        Path to save the plot.
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(properties_list)
    
    # Select properties if specified
    if property_names:
        df = df[property_names]
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Matrix of Molecular Properties')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_property_scatter(properties_list, x_property, y_property, color_property=None, save_path=None):
    """
    Plot scatter plot of two molecular properties.
    
    Parameters
    ----------
    properties_list : list
        List of property dictionaries.
    x_property : str
        Name of the property for x-axis.
    y_property : str
        Name of the property for y-axis.
    color_property : str, optional
        Name of the property for color.
    save_path : str, optional
        Path to save the plot.
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(properties_list)
    
    plt.figure(figsize=(10, 8))
    
    if color_property:
        scatter = plt.scatter(
            df[x_property], 
            df[y_property], 
            c=df[color_property], 
            cmap='viridis', 
            alpha=0.7
        )
        plt.colorbar(scatter, label=color_property)
    else:
        plt.scatter(df[x_property], df[y_property], alpha=0.7)
    
    plt.title(f'{y_property} vs {x_property}')
    plt.xlabel(x_property)
    plt.ylabel(y_property)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_molecule_with_atom_indices(smiles, molSize=(400, 300)):
    """
    Visualize a molecule with atom indices.
    
    Parameters
    ----------
    smiles : str
        SMILES string.
    molSize : tuple, optional (default=(400, 300))
        Size of the molecule image.
        
    Returns
    -------
    PIL.Image
        Image of the molecule with atom indices.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    # Generate 2D coordinates if not available
    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)
    
    # Add atom indices
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    
    # Draw molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert to PIL image
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    
    plt.figure(figsize=(molSize[0]/100, molSize[1]/100))
    plt.imshow(img)
    plt.title("Molecule with Atom Indices")
    plt.axis('off')
    plt.show()
    
    return img 