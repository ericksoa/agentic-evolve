"""
3D Conformer Generation for hERG Prediction

Generates 3D molecular conformers from SMILES using RDKit's ETKDG method.
Conformers are optimized with MMFF94 force field for more realistic geometries.

Used by:
- embeddings_3d.py: Input for Uni-Mol2 3D embeddings
- visualize_molecules.py: 3D molecular visualization
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from typing import Optional, Tuple, Dict, List
import warnings

# Suppress RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def generate_conformer(
    smiles: str,
    num_conformers: int = 1,
    random_seed: int = 42,
    max_attempts: int = 100,
    optimize: bool = True
) -> Optional[Chem.Mol]:
    """
    Generate 3D conformer(s) for a molecule from SMILES.

    Args:
        smiles: SMILES string of the molecule
        num_conformers: Number of conformers to generate
        random_seed: Random seed for reproducibility
        max_attempts: Maximum embedding attempts
        optimize: Whether to optimize with MMFF94 force field

    Returns:
        RDKit Mol object with 3D coordinates, or None if generation fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens for better 3D structure
    mol = Chem.AddHs(mol)

    # ETKDG parameters for high-quality conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.maxAttempts = max_attempts
    params.numThreads = 1  # Single-threaded for reproducibility
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    # Generate conformers
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

    if len(conf_ids) == 0:
        # Fallback: try without ETKDG constraints
        params = AllChem.ETKDGv2()
        params.randomSeed = random_seed
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

        if len(conf_ids) == 0:
            return None

    # Optimize with force field if requested
    if optimize:
        for conf_id in conf_ids:
            try:
                # Try MMFF94 first (more accurate)
                result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                if result != 0:
                    # Fallback to UFF
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
            except Exception:
                # If optimization fails, keep unoptimized conformer
                pass

    return mol


def get_3d_descriptors(mol: Chem.Mol, conf_id: int = 0) -> Dict[str, float]:
    """
    Extract 3D molecular descriptors from a conformer.

    These descriptors capture shape, size, and spatial properties
    relevant to hERG channel binding.

    Args:
        mol: RDKit Mol object with 3D coordinates
        conf_id: Conformer ID to use

    Returns:
        Dictionary of 3D descriptor names and values
    """
    if mol is None or mol.GetNumConformers() == 0:
        return _get_zero_descriptors()

    try:
        conf = mol.GetConformer(conf_id)
    except Exception:
        return _get_zero_descriptors()

    descriptors = {}

    # Shape descriptors - Principal Moments of Inertia
    try:
        pmi1, pmi2, pmi3 = Descriptors3D.PMI1(mol, confId=conf_id), \
                           Descriptors3D.PMI2(mol, confId=conf_id), \
                           Descriptors3D.PMI3(mol, confId=conf_id)
        descriptors['PMI1'] = pmi1
        descriptors['PMI2'] = pmi2
        descriptors['PMI3'] = pmi3

        # Normalized PMI ratios (shape indicators)
        if pmi3 > 0:
            descriptors['NPR1'] = pmi1 / pmi3  # Ranges 0-1, rod-like to spherical
            descriptors['NPR2'] = pmi2 / pmi3
        else:
            descriptors['NPR1'] = 0.0
            descriptors['NPR2'] = 0.0
    except Exception:
        descriptors.update({'PMI1': 0, 'PMI2': 0, 'PMI3': 0, 'NPR1': 0, 'NPR2': 0})

    # Asphericity and Eccentricity
    try:
        descriptors['Asphericity'] = Descriptors3D.Asphericity(mol, confId=conf_id)
        descriptors['Eccentricity'] = Descriptors3D.Eccentricity(mol, confId=conf_id)
        descriptors['InertialShapeFactor'] = Descriptors3D.InertialShapeFactor(mol, confId=conf_id)
    except Exception:
        descriptors.update({'Asphericity': 0, 'Eccentricity': 0, 'InertialShapeFactor': 0})

    # Radius of Gyration - molecular size
    try:
        descriptors['RadiusOfGyration'] = Descriptors3D.RadiusOfGyration(mol, confId=conf_id)
    except Exception:
        descriptors['RadiusOfGyration'] = 0

    # Spherocity Index - how sphere-like
    try:
        descriptors['SpherocityIndex'] = Descriptors3D.SpherocityIndex(mol, confId=conf_id)
    except Exception:
        descriptors['SpherocityIndex'] = 0

    # Plane of Best Fit RMS - molecular planarity
    try:
        descriptors['PBF'] = rdMolDescriptors.CalcPBF(mol, confId=conf_id)
    except Exception:
        descriptors['PBF'] = 0

    return descriptors


def get_pharmacophore_distances(mol: Chem.Mol, conf_id: int = 0) -> Dict[str, float]:
    """
    Calculate distances between key pharmacophore features for hERG.

    hERG blockers typically have:
    - Basic nitrogen (protonated at physiological pH)
    - Aromatic rings for π-stacking
    - Hydrophobic regions

    Args:
        mol: RDKit Mol object with 3D coordinates
        conf_id: Conformer ID to use

    Returns:
        Dictionary of pharmacophore distance features
    """
    if mol is None or mol.GetNumConformers() == 0:
        return _get_zero_pharmacophore()

    try:
        conf = mol.GetConformer(conf_id)
    except Exception:
        return _get_zero_pharmacophore()

    features = {}

    # Find basic nitrogens (sp3 N with H, potential protonation sites)
    basic_n_positions = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:  # Nitrogen
            # Check if basic (sp3 with H neighbors or positive charge)
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                pos = conf.GetAtomPosition(atom.GetIdx())
                basic_n_positions.append(np.array([pos.x, pos.y, pos.z]))

    # Find aromatic ring centroids
    aromatic_centroids = []
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            centroid = np.zeros(3)
            for idx in ring:
                pos = conf.GetAtomPosition(idx)
                centroid += np.array([pos.x, pos.y, pos.z])
            centroid /= len(ring)
            aromatic_centroids.append(centroid)

    # Calculate distances
    features['num_basic_N'] = len(basic_n_positions)
    features['num_aromatic_rings'] = len(aromatic_centroids)

    # Basic N to aromatic centroid distances
    if basic_n_positions and aromatic_centroids:
        distances = []
        for n_pos in basic_n_positions:
            for ar_pos in aromatic_centroids:
                distances.append(np.linalg.norm(n_pos - ar_pos))
        features['min_N_aromatic_dist'] = min(distances)
        features['max_N_aromatic_dist'] = max(distances)
        features['mean_N_aromatic_dist'] = np.mean(distances)
    else:
        features['min_N_aromatic_dist'] = 0
        features['max_N_aromatic_dist'] = 0
        features['mean_N_aromatic_dist'] = 0

    # Aromatic ring to aromatic ring distances
    if len(aromatic_centroids) >= 2:
        ar_distances = []
        for i, ar1 in enumerate(aromatic_centroids):
            for ar2 in aromatic_centroids[i+1:]:
                ar_distances.append(np.linalg.norm(ar1 - ar2))
        features['min_aromatic_dist'] = min(ar_distances)
        features['max_aromatic_dist'] = max(ar_distances)
        features['mean_aromatic_dist'] = np.mean(ar_distances)
    else:
        features['min_aromatic_dist'] = 0
        features['max_aromatic_dist'] = 0
        features['mean_aromatic_dist'] = 0

    # Molecular extent (max distance between any two atoms)
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append(np.array([pos.x, pos.y, pos.z]))

    if len(positions) >= 2:
        max_dist = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                d = np.linalg.norm(p1 - p2)
                if d > max_dist:
                    max_dist = d
        features['molecular_extent'] = max_dist
    else:
        features['molecular_extent'] = 0

    return features


def _get_zero_descriptors() -> Dict[str, float]:
    """Return zero values for all 3D descriptors (for failed molecules)."""
    return {
        'PMI1': 0, 'PMI2': 0, 'PMI3': 0,
        'NPR1': 0, 'NPR2': 0,
        'Asphericity': 0, 'Eccentricity': 0, 'InertialShapeFactor': 0,
        'RadiusOfGyration': 0, 'SpherocityIndex': 0, 'PBF': 0
    }


def _get_zero_pharmacophore() -> Dict[str, float]:
    """Return zero values for all pharmacophore features (for failed molecules)."""
    return {
        'num_basic_N': 0, 'num_aromatic_rings': 0,
        'min_N_aromatic_dist': 0, 'max_N_aromatic_dist': 0, 'mean_N_aromatic_dist': 0,
        'min_aromatic_dist': 0, 'max_aromatic_dist': 0, 'mean_aromatic_dist': 0,
        'molecular_extent': 0
    }


def get_all_3d_features(smiles: str, random_seed: int = 42) -> Dict[str, float]:
    """
    Extract all 3D features for a molecule.

    Combines shape descriptors and pharmacophore distances into
    a single feature dictionary.

    Args:
        smiles: SMILES string of the molecule
        random_seed: Random seed for conformer generation

    Returns:
        Dictionary of all 3D features (20 features total)
    """
    mol = generate_conformer(smiles, num_conformers=1, random_seed=random_seed)

    features = {}
    features.update(get_3d_descriptors(mol))
    features.update(get_pharmacophore_distances(mol))

    return features


def get_3d_feature_names() -> List[str]:
    """Return list of all 3D feature names in consistent order."""
    return [
        # Shape descriptors (11)
        'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
        'Asphericity', 'Eccentricity', 'InertialShapeFactor',
        'RadiusOfGyration', 'SpherocityIndex', 'PBF',
        # Pharmacophore distances (9)
        'num_basic_N', 'num_aromatic_rings',
        'min_N_aromatic_dist', 'max_N_aromatic_dist', 'mean_N_aromatic_dist',
        'min_aromatic_dist', 'max_aromatic_dist', 'mean_aromatic_dist',
        'molecular_extent'
    ]


if __name__ == '__main__':
    # Test conformer generation
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)NCC(O)C1=CC=C(O)C(O)=C1',  # Isoproterenol (hERG blocker)
    ]

    print("Testing 3D conformer generation and feature extraction:")
    print("=" * 60)

    for smi in test_smiles:
        print(f"\nSMILES: {smi}")
        features = get_all_3d_features(smi)
        print(f"  3D Features extracted: {len(features)}")
        print(f"  Radius of Gyration: {features['RadiusOfGyration']:.2f}")
        print(f"  Asphericity: {features['Asphericity']:.3f}")
        print(f"  Basic N count: {features['num_basic_N']}")
        print(f"  Aromatic rings: {features['num_aromatic_rings']}")
        if features['min_N_aromatic_dist'] > 0:
            print(f"  N-aromatic distance: {features['min_N_aromatic_dist']:.2f} Å")
