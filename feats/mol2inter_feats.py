import re
import sys
import time

import numpy as np
import torch
import rdkit.Chem as Chem
from rdkit import RDLogger
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

np.set_printoptions(suppress=True, precision=8)
np.seterr(all='raise')
np.float_ = np.float32

from shielded_detection import filter_interactions_los
from atom_feats import prot_lig_pair_feats, lig_lig_pair_feats
from atom_pairs import extract_atom_pairs


def get_position(mol):
	if mol.GetNumConformers() == 0:
		raise ValueError("No conformers found for the molecule.")
	return mol.GetConformer().GetPositions()


def load_molecule(file_path):
    """
    Load a molecule from PDB, MOL2, or SDF file with relaxed sanitization.

    Disables RDKit logging during loading and applies partial sanitization
    (skipping kekulization and property assignment for compatibility with
    non-standard chemical structures). For PDB files without CONECT records,
    manually reconstructs the conformer from ATOM/HETATM coordinate lines.
    """
    mol = None
    RDLogger.DisableLog('rdApp.*')
    try:
        if re.search(r'\.pdb', file_path, re.I):
            mol = Chem.MolFromPDBFile(file_path, sanitize=False, removeHs=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, 
                                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                        Chem.SanitizeFlags.SANITIZE_KEKULIZE ^
                                        Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except:
                    pass
        elif re.search(r'\.mol2', file_path, re.I):
            mol = Chem.MolFromMol2File(file_path, sanitize=False, removeHs=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, 
                                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                        Chem.SanitizeFlags.SANITIZE_KEKULIZE ^
                                        Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except:
                    pass
        elif re.search(r'\.sdf', file_path, re.I):
            suppl = Chem.SDMolSupplier(file_path, sanitize=False, removeHs=False)
            mol = next(iter(suppl))
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, 
                                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                        Chem.SanitizeFlags.SANITIZE_KEKULIZE ^
                                        Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except:
                    pass
        else:
            raise ValueError("Unsupported file format")
    finally:
        RDLogger.EnableLog('rdApp.*')
    
    if mol is None:
        raise ValueError(f"Unable to load molecule from file: {file_path}")
    
    if mol.GetNumConformers() == 0:
        if re.search(r'\.pdb', file_path, re.I):
            conf = Chem.Conformer(mol.GetNumAtoms())
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            try:
                                atom_idx = int(line[6:11]) - 1
                                x = float(line[30:38])
                                y = float(line[38:46])
                                z = float(line[46:54])
                                conf.SetAtomPosition(atom_idx, (x, y, z))
                            except:
                                continue
                mol.AddConformer(conf)
            except:
                pass
    return mol


def read_prot_lig(mol_p, mol_l, pocket_cutoff=12.0):
    # Get protein and ligand coordinates
    coords_p = get_position(mol_p)
    coords_l = get_position(mol_l)
    prot_tree = cKDTree(coords_p)
    indices = prot_tree.query_ball_point(coords_l, r=pocket_cutoff)
    pocket_atom_indices = list(set(index for sublist in indices for index in sublist))
    coords_p = coords_p[pocket_atom_indices]
    return pocket_atom_indices, coords_p, coords_l


def mol_to_distance_matrix(coords_p, coords_l, D_cutoff=12.0):
    # Combine coordinates and calculate distance matrix
    all_coords = np.vstack((coords_p, coords_l))
    distance_matrix = cdist(all_coords, all_coords)
    distance_matrix[distance_matrix > D_cutoff] = 12.0
    num_p = len(coords_p)
    return distance_matrix, num_p


def create_protein_ligand_mask(num_p, num_l):
    # Create mask for protein-ligand interactions
    total_atoms = num_p + num_l
    mask = np.ones((total_atoms, total_atoms)) * np.nan
    mask[num_p:, :num_p] = 1
    mask[:num_p, num_p:] = 1
    return mask


def filter_matrix(matrix, num_p, num_l):
    # Apply protein-ligand mask to matrix
    mask = create_protein_ligand_mask(num_p, num_l)
    return matrix * mask


def extract_pl_matirx(matrix, num_p):
    # Extract protein-ligand interaction block
    pl_matirx = matrix[num_p:, :num_p]
    return pl_matirx


def extract_valid_pairs(matrix, pairs_mask):
    # Extract valid atom pairs from matrix
    tensor = torch.from_numpy(matrix)
    pairs_mask = torch.tensor(pairs_mask, dtype=torch.bool)
    
    if len(tensor.shape) == 2:
        valid_pairs_flat = pairs_mask.flatten()
        return tensor.flatten()[valid_pairs_flat].view(-1, 1)
    elif len(tensor.shape) == 3:
        valid_pairs_flat = pairs_mask.flatten()
        return tensor.reshape(-1, tensor.shape[-1])[valid_pairs_flat]


def gen_imask(distance_matrix, prot_indices, mol_p, mol_l,
              pairs_cutoff=5.0, tol_cutoff=-0.5, buffer_dist=0.5,
              is_inference=False):
    """Generate a binary protein-ligand interaction mask via distance cutoff, water exclusion, and line-of-sight filtering."""
    num_p = len(prot_indices)
    total_atoms = distance_matrix.shape[0]
    num_l = total_atoms - num_p
    
    # Initialize mask for protein-ligand pairs
    mask = np.zeros((total_atoms, total_atoms))
    mask[num_p:, :num_p] = 1
    mask[:num_p, num_p:] = 1
    
    # Filter pairs by distance cutoff
    pairs_mask = distance_matrix <= pairs_cutoff
    mask_pairs_cutoff = mask * pairs_cutoff
    
    # Remove water molecules if not in inference mode
    if not is_inference:
        water_atoms = []
        for i in range(num_p):
            prot_atom = mol_p.GetAtomWithIdx(prot_indices[i])
            if (prot_atom.GetPDBResidueInfo() and 
                prot_atom.GetPDBResidueInfo().GetResidueName() == "HOH" and
                prot_atom.GetSymbol() in ["H", "O"]):
                water_atoms.append(i)
        
        if water_atoms:
            water_indices = np.array(water_atoms)
            clash_mask = distance_matrix[water_indices, num_p:] < 2.4
            has_clash = np.any(clash_mask, axis=1)
            for i, idx in enumerate(water_indices[has_clash]):
                pairs_mask[idx, :] = False
                pairs_mask[:, idx] = False

    # Get all possible pairs and filter by chemical interactions
    mask_pairs_cutoff = mask * pairs_mask
    prob_pairs = np.argwhere(mask_pairs_cutoff)
    all_pairs = [(prot_indices[i], j - num_p) for i, j in prob_pairs if i < num_p and j >= num_p]

    filtered_pairs = filter_interactions_los(mol_p, mol_l, all_pairs, 
                                          tol_cutoff=tol_cutoff, buffer_dist=buffer_dist)

    # Create final mask with filtered pairs
    final_mask = np.zeros_like(mask_pairs_cutoff)
    for p_idx, l_idx in filtered_pairs:
        i = prot_indices.index(p_idx)
        j = l_idx + num_p
        final_mask[i, j] = final_mask[j, i] = 1
    
    final_mask = filter_matrix(final_mask, num_p, num_l)
    return final_mask


def cplx_to_interactions(prot_file_path, lig_file_path, is_inference=True):
    """Convert a protein-ligand complex into atomic-pair interaction features: pocket extraction, LOS filtering, atom-pair encoding, and ligand rotatable-bond pair extraction."""
    # Load molecules
    mol_p = load_molecule(prot_file_path)
    mol_l = load_molecule(lig_file_path)
    
    # Generate pocket and coordinates
    prot_indices, coords_p, coords_l = read_prot_lig(mol_p, mol_l, pocket_cutoff=10.0)
    
    # Generate distance matrix and extract protein-ligand region
    distance_matrix, num_p = mol_to_distance_matrix(coords_p, coords_l, D_cutoff=12.0)
    pl_dist = extract_pl_matirx(distance_matrix, num_p)
    
    # Generate protein-ligand interaction mask
    pl_pairs_mask = gen_imask(distance_matrix, 
                            prot_indices, 
                            mol_p, 
                            mol_l, 
                            pairs_cutoff=5.0,
                            tol_cutoff=-0.5,
                            buffer_dist=0.5,
                            is_inference=is_inference)
    
    # Extract protein-ligand interaction region
    pl_pairs_mask = extract_pl_matirx(pl_pairs_mask, num_p)
    pl_dist = extract_valid_pairs(pl_dist, pl_pairs_mask).float()
    
    # Encode complex atomic pair features
    prot_atom_feats, lig_atom_feats, prot_neigh, lig_neigh, pl_angles = prot_lig_pair_feats(
        mol_p, mol_l, prot_indices, pl_pairs_mask)
    
    # Extract ligand internal atom pairs based on rotatable bonds
    ll_pairs_idx, ll_dist = extract_atom_pairs(mol_l, step_deg=15, clash_threshold=1.1)
    original_pairs_to_dist = {pair: dist for pair, dist in zip(ll_pairs_idx, ll_dist)}
    if not ll_pairs_idx:
        if prot_atom_feats.numel() > 0:
            atom_feat_dim = prot_atom_feats.shape[1]
            neighbor_feat_dim = prot_neigh.shape[2]

        # Create empty tensors
        ll_dist = torch.empty((0, 1), dtype=torch.float32)
        ll_atom1_feats = torch.empty((0, atom_feat_dim), dtype=torch.float32)
        ll_atom2_feats = torch.empty((0, atom_feat_dim), dtype=torch.float32)
        ll_neigh1 = torch.empty((0, 6, neighbor_feat_dim), dtype=torch.float32)
        ll_neigh2 = torch.empty((0, 6, neighbor_feat_dim), dtype=torch.float32)
    else:
        # Get distances for filtered atom pairs
        ll_dist = torch.stack([original_pairs_to_dist[pair] for pair in ll_pairs_idx])
        # Encode ligand pair features
        ll_atom1_feats, ll_atom2_feats, ll_neigh1, ll_neigh2 = lig_lig_pair_feats(mol_l, ll_pairs_idx)
    
    return pl_dist, prot_atom_feats, lig_atom_feats, prot_neigh, lig_neigh, pl_angles, \
           ll_dist, ll_atom1_feats, ll_atom2_feats, ll_neigh1, ll_neigh2




