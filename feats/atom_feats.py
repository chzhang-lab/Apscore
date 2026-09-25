import numpy as np
import torch
from rdkit import Chem
from constants import RCSD_VDW_DATA

def get_vdw_radius(atom):
	"""Get van der Waals radius for an atom."""
	return RCSD_VDW_DATA[atom.GetAtomicNum() - 1] if atom.GetAtomicNum() <= len(RCSD_VDW_DATA) else 2.0

class AtomEncoder:
	"""
	Atom encoder with pre-computed trigonometric tables.
	"""
	_instance = None
	
	def __init__(self):
		if AtomEncoder._instance is not None:
			return
		
		self.symbol_map = {sym: idx for idx, sym in enumerate([
			'H', 'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I',
			'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Na', 'K',
			'Ca', 'Mg', 'Se', 'other'
		])}
		self.hybrid_map = {hyb: idx for idx, hyb in enumerate([
			Chem.rdchem.HybridizationType.SP,
			Chem.rdchem.HybridizationType.SP2,
			Chem.rdchem.HybridizationType.SP3,
			Chem.rdchem.HybridizationType.SP3D,
			Chem.rdchem.HybridizationType.SP3D2,
			'other'
		])}
		
		self.chirality_map = {chiral: idx for idx, chiral in enumerate([
			Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,  # R
			Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, # S
			'other'
		])}
		
		x = np.arange(0, 1.01, 0.01) * np.pi
		self.sin_table = np.sin(x)
		self.cos_table = np.cos(x)
		
		AtomEncoder._instance = self
	
	@classmethod
	def get_instance(cls):
		"""Get singleton instance of AtomEncoder."""
		if cls._instance is None:
			cls._instance = cls()
		return cls._instance
	
	def _is_amide_nitrogen(self, atom):
		"""Check if atom is an amide nitrogen."""
		if atom.GetSymbol() != 'N':
			return False
		
		# Check if connected to carbonyl carbon
		for neighbor in atom.GetNeighbors():
			if neighbor.GetSymbol() == 'C':
				# Check if this carbon is connected to oxygen 
				for carbon_neighbor in neighbor.GetNeighbors():
					if carbon_neighbor.GetSymbol() == 'O':
						bond = atom.GetOwningMol().GetBondBetweenAtoms(
							atom.GetIdx(), neighbor.GetIdx()
						)
						if bond and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
							return True
		return False
	
	def _get_nh_donor_type(self, atom):
		"""Get N-H donor type (normalized to [0,1])."""
		if atom.GetSymbol() != 'N':
			return 0.0
		total_h = atom.GetTotalNumHs()
		if total_h == 0:
			return 0.0
		
		# Check if in amide
		if self._is_amide_nitrogen(atom):
			return 3.0 / 6.0
		
		# Check if aromatic nitrogen
		if atom.GetIsAromatic():
			return 4.0 / 6.0 
		
		# Check charge state
		if atom.GetFormalCharge() > 0:
			return 5.0 / 6.0
		
		# Determine primary/secondary amine based on degree
		degree = atom.GetDegree()
		if degree == 1: 
			return 1.0 / 6.0  
		elif degree == 2:
			return 2.0 / 6.0  
		
		return 0.0  # Other types
	
	def _calculate_hbond_strength(self, atom, donor=True):
		"""Calculate hydrogen bond strength (normalized to [0,1])."""
		symbol = atom.GetSymbol()
		if donor:
			if symbol == 'O':
				return 1.0
			elif symbol == 'N':
				return 0.8
			elif symbol == 'S':
				return 0.6
			else:
				return 0.0
		else:
			strength_map = {'O': 1.0, 'N': 0.9, 'F': 0.8, 
						'Cl': 0.7, 'Br': 0.6, 'I': 0.5, 'S': 0.4}
			return strength_map.get(symbol, 0.0)
	
	def _get_local_environment(self, atom):
		"""Get local environment features (normalized to [0,1])."""
		electronegative_neighbors = 0
		for neighbor in atom.GetNeighbors():
			if neighbor.GetSymbol() in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
				electronegative_neighbors += 1
	
		electronegative_env = electronegative_neighbors / 6.0  # Max neighbors
		# Ring constraint feature
		ring_constraint = 1.0 if atom.IsInRing() else 0.0
		
		return electronegative_env, ring_constraint
	
	def _get_geometry_accessibility(self, atom):
		"""Get geometry accessibility features (normalized to [0,1])."""
		degree = atom.GetDegree()
		if degree == 0:
			return 1.0  
		elif degree == 1:
			return 0.8
		elif degree == 2:
			return 0.6
		elif degree == 3:
			return 0.4
		else:
			return 0.2

	def _get_hbond_features(self, atom):
		"""
		Get hydrogen bonding features (8-dimensional, normalized to [0,1]).
		
		Args:
			atom: RDKit atom object
			
		Returns:
			list: List of 8 hydrogen bonding features
		"""
		symbol = atom.GetSymbol()
		total_h = atom.GetTotalNumHs()
		
		# Basic hydrogen bond donor capability
		hbond_donor = 0.0
		if symbol in ['N', 'O', 'S'] and total_h > 0:
			hbond_donor = 1.0
		
		# Basic hydrogen bond acceptor capability
		hbond_acceptor = 0.0
		if symbol in ['N', 'O', 'F', 'Cl', 'Br', 'I', 'S']:
			# Exclude positively charged nitrogen (weak acceptor)
			if symbol == 'N' and atom.GetFormalCharge() > 0:
				hbond_acceptor = 0.5
			else:
				hbond_acceptor = 1.0

		donor_strength = self._calculate_hbond_strength(atom, donor=True)
		acceptor_strength = self._calculate_hbond_strength(atom, donor=False)
		nh_donor_type = self._get_nh_donor_type(atom)
		electronegative_env, ring_constraint = self._get_local_environment(atom)
		geometry_accessibility = self._get_geometry_accessibility(atom)
		
		return [
			hbond_donor,
			hbond_acceptor,
			donor_strength,
			acceptor_strength,
			nh_donor_type,
			electronegative_env,
			ring_constraint,
			geometry_accessibility
		]

	def encode(self, atom):
		"""Encode an atom as a 23-dim feature vector: 14 trig-encoded properties + aromatic flag + 8 H-bond descriptors."""
		continuous_features = [
			self.symbol_map.get(atom.GetSymbol(), self.symbol_map['other']) / len(self.symbol_map),
			atom.GetDegree() / 7.0,
			atom.GetFormalCharge() / 5.0,
			atom.GetNumRadicalElectrons() / 4.0,
			self.hybrid_map.get(atom.GetHybridization(), self.hybrid_map['other']) / len(self.hybrid_map),
			get_vdw_radius(atom) / 3.0,
			self.chirality_map.get(atom.GetChiralTag(), self.chirality_map['other']) / len(self.chirality_map)
		]
		encoded = []
		for feat in continuous_features:
			idx = int(feat * 100)
			idx = max(0, min(100, idx))  
			encoded.extend([self.sin_table[idx], self.cos_table[idx]])
		
		encoded.append(float(atom.GetIsAromatic()))
		hbond_features = self._get_hbond_features(atom)
		encoded.extend(hbond_features)
		
		return np.array(encoded, dtype=np.float32) 

def get_bond_type_encoding(atom, neighbor):
	bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
	bond_map = {
		Chem.rdchem.BondType.SINGLE: 1,
		Chem.rdchem.BondType.DOUBLE: 2,
		Chem.rdchem.BondType.TRIPLE: 3,
		Chem.rdchem.BondType.AROMATIC: 4
	}
	stereo_map = {
		Chem.rdchem.BondStereo.STEREONONE: 1,
		Chem.rdchem.BondStereo.STEREOANY: 2,
		Chem.rdchem.BondStereo.STEREOZ: 3,
		Chem.rdchem.BondStereo.STEREOE: 4
	}
	bond_type_norm = bond_map.get(bond.GetBondType(), 0) / 4.0
	stereo_norm = stereo_map.get(bond.GetStereo(), 0) / 4.0
	
	encoder = AtomEncoder.get_instance()
	idx1 = int(bond_type_norm * 100)
	idx2 = int(stereo_norm * 100)
	idx1 = max(0, min(100, idx1))
	idx2 = max(0, min(100, idx2))
	
	encoded = [
		encoder.sin_table[idx1], encoder.cos_table[idx1],
		encoder.sin_table[idx2], encoder.cos_table[idx2],
		float(bond.GetIsConjugated()),
		float(bond.IsInRing())
	]
	
	return np.array(encoded, dtype=np.float32)

def get_atom_features(atom):
	encoder = AtomEncoder.get_instance()
	return encoder.encode(atom)

def get_neighbor_features(atom):
	max_neighbors = 6
	atom_feat_dim = 23
	bond_feat_dim = 6
	feat_dim = atom_feat_dim + bond_feat_dim
	
	neighbors = list(atom.GetNeighbors())
	num_neighbors = min(len(neighbors), max_neighbors)
	
	if num_neighbors == 0:
		return np.zeros((max_neighbors, feat_dim), dtype=np.float32)
	
	# Pre-allocate array
	neighbor_feats = np.zeros((num_neighbors, feat_dim), dtype=np.float32)
	
	for i in range(num_neighbors):
		neighbor = neighbors[i]
		neighbor_feats[i, :atom_feat_dim] = get_atom_features(neighbor)
		neighbor_feats[i, atom_feat_dim:] = get_bond_type_encoding(atom, neighbor)
	
	# Pad to maximum neighbors
	if num_neighbors < max_neighbors:
		full_feats = np.zeros((max_neighbors, feat_dim), dtype=np.float32)
		full_feats[:num_neighbors] = neighbor_feats
		return full_feats
	
	return neighbor_feats

def calculate_average_position(mol, atom_list, current_atom_idx=None):
	"""
	Calculate average position of atoms in the list.
	"""
	if not atom_list:
		if current_atom_idx is not None:
			atom_idx = int(current_atom_idx)
			pos = mol.GetConformer().GetAtomPosition(atom_idx)
			return np.array([pos.x, pos.y, pos.z])
		else:
			return np.array([10.0, 10.0, 10.0])
	
	conf = mol.GetConformer()
	positions = []
	for nbr in atom_list:
		pos = conf.GetAtomPosition(int(nbr.GetIdx()))
		positions.append([pos.x, pos.y, pos.z])
	
	return np.mean(positions, axis=0)

def calculate_interaction_angle(mol_p, mol_l, prot_indices, pairs_mask):
	"""
	Calculate interaction angles between protein and ligand atoms.
	"""
	conf_p = mol_p.GetConformer()
	conf_l = mol_l.GetConformer()
	valid_pairs = np.argwhere(pairs_mask == 1)
	
	cos_angles = np.zeros(len(valid_pairs), dtype=np.float32)
	
	for idx, (l_idx, p_idx) in enumerate(valid_pairs):
		prot_atom_idx = int(prot_indices[int(p_idx)])
		lig_atom_idx = int(l_idx)
		
		# Safety check: ensure valid atom indices
		if (prot_atom_idx < 0 or prot_atom_idx >= mol_p.GetNumAtoms() or
			lig_atom_idx < 0 or lig_atom_idx >= mol_l.GetNumAtoms()):
			cos_angles[idx] = 0.0
			continue
		
		# Get atoms
		lig_atom = mol_l.GetAtomWithIdx(lig_atom_idx)
		prot_atom = mol_p.GetAtomWithIdx(prot_atom_idx)
		
		# Get neighbors
		neighbors_prot = list(prot_atom.GetNeighbors())
		neighbors_lig = list(lig_atom.GetNeighbors())
		
		# Calculate average positions
		c_position = calculate_average_position(mol_p, neighbors_prot, prot_atom_idx)
		d_position = calculate_average_position(mol_l, neighbors_lig, lig_atom_idx)
		
		# Get atom positions
		pos_a = conf_p.GetAtomPosition(prot_atom_idx)
		pos_b = conf_l.GetAtomPosition(lig_atom_idx)
		a_position = np.array([pos_a.x, pos_a.y, pos_a.z])
		b_position = np.array([pos_b.x, pos_b.y, pos_b.z])
		
		# Calculate vectors
		vector_ca = a_position - c_position
		vector_db = b_position - d_position
		
		norm_ca = np.linalg.norm(vector_ca)
		norm_db = np.linalg.norm(vector_db)
		
		if norm_ca < 1e-6 or norm_db < 1e-6:
			angle = 0.0
		else:
			cos_angle = np.dot(vector_ca, vector_db) / (norm_ca * norm_db)
			cos_angle = max(min(cos_angle, 1.0), -1.0)
			angle = np.degrees(np.arccos(cos_angle)) / 100
		
		cos_angles[idx] = angle
	
	return cos_angles


def prot_lig_pair_feats(mol_p, mol_l, prot_indices, pairs_mask):
    """
    Extract features for protein-lignad atom pairs.

    """
    prot_atom_feats = []
    lig_atom_feats = []
    prot_neighbor_feats = []
    lig_neighbor_feats = []
    angle_feats = []
    
    # Calculate interaction angles
    cos_angles = calculate_interaction_angle(mol_p, mol_l, prot_indices, pairs_mask)
    valid_pairs = np.argwhere(pairs_mask == 1)
    
    for idx, (l_idx, p_idx) in enumerate(valid_pairs):
        prot_atom_idx = int(prot_indices[int(p_idx)])
        lig_atom_idx = int(l_idx)
        
        # Safety check
        if (prot_atom_idx < 0 or prot_atom_idx >= mol_p.GetNumAtoms() or
            lig_atom_idx < 0 or lig_atom_idx >= mol_l.GetNumAtoms()):
            # Use zero features for invalid indices
            prot_feat = np.zeros(23, dtype=np.float32)
            lig_feat = np.zeros(23, dtype=np.float32)
            prot_neighbors = np.zeros((6, 29), dtype=np.float32)
            lig_neighbors = np.zeros((6, 29), dtype=np.float32)
        else:
            prot_atom = mol_p.GetAtomWithIdx(prot_atom_idx)
            lig_atom = mol_l.GetAtomWithIdx(lig_atom_idx)
            
            # Extract features
            prot_feat = get_atom_features(prot_atom)
            lig_feat = get_atom_features(lig_atom)
            
            prot_neighbors = get_neighbor_features(prot_atom)
            lig_neighbors = get_neighbor_features(lig_atom)
        
        prot_atom_feats.append(prot_feat)
        lig_atom_feats.append(lig_feat)
        prot_neighbor_feats.append(prot_neighbors)
        lig_neighbor_feats.append(lig_neighbors)
        angle_feats.append(cos_angles[idx])
    
    # Convert to tensors
    prot_atom_feats = torch.tensor(np.array(prot_atom_feats))
    lig_atom_feats = torch.tensor(np.array(lig_atom_feats))
    prot_neighbor_feats = torch.tensor(np.array(prot_neighbor_feats))
    lig_neighbor_feats = torch.tensor(np.array(lig_neighbor_feats))
    angle_feats = torch.tensor(np.array(angle_feats)).view(-1, 1)
    
    return prot_atom_feats, lig_atom_feats, prot_neighbor_feats, lig_neighbor_feats, angle_feats

def lig_lig_pair_feats(mol_l, ligand_pairs):
    """
    Extract features for ligand-lignad atom pairs.
    """
    lig_atom1_feats = []
    lig_atom2_feats = []
    lig_neighbor1_feats = []
    lig_neighbor2_feats = []
    
    # Extract features for each atom pair
    for idx1, idx2 in ligand_pairs:
        idx1_int = int(idx1)
        idx2_int = int(idx2)
        
        # Safety check
        if (idx1_int < 0 or idx1_int >= mol_l.GetNumAtoms() or
            idx2_int < 0 or idx2_int >= mol_l.GetNumAtoms()):
            # Use zero features for invalid indices
            lig_feat1 = np.zeros(23, dtype=np.float32)
            lig_feat2 = np.zeros(23, dtype=np.float32)
            lig_neighbors1 = np.zeros((6, 29), dtype=np.float32)
            lig_neighbors2 = np.zeros((6, 29), dtype=np.float32)
        else:
            # Get the two atoms
            lig_atom1 = mol_l.GetAtomWithIdx(idx1_int)
            lig_atom2 = mol_l.GetAtomWithIdx(idx2_int)
            
            # Extract atom features
            lig_feat1 = get_atom_features(lig_atom1)
            lig_feat2 = get_atom_features(lig_atom2)
            
            # Extract neighbor features
            lig_neighbors1 = get_neighbor_features(lig_atom1)
            lig_neighbors2 = get_neighbor_features(lig_atom2)
        
        # Add to result lists
        lig_atom1_feats.append(lig_feat1)
        lig_atom2_feats.append(lig_feat2)
        lig_neighbor1_feats.append(lig_neighbors1)
        lig_neighbor2_feats.append(lig_neighbors2)
    
    # Convert to tensors
    lig_atom1_feats = torch.tensor(np.array(lig_atom1_feats))
    lig_atom2_feats = torch.tensor(np.array(lig_atom2_feats))
    lig_neighbor1_feats = torch.tensor(np.array(lig_neighbor1_feats))
    lig_neighbor2_feats = torch.tensor(np.array(lig_neighbor2_feats))
    
    return lig_atom1_feats, lig_atom2_feats, lig_neighbor1_feats, lig_neighbor2_feats
