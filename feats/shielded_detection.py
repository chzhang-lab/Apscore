import numpy as np
from scipy.spatial import cKDTree
from rdkit import Chem
from typing import List, Tuple

from constants import RCSD_VDW_DATA

def get_vdw_radii(mol: Chem.Mol) -> np.ndarray:
	return np.array([RCSD_VDW_DATA[atom.GetAtomicNum() - 1] 
					if atom.GetAtomicNum() <= len(RCSD_VDW_DATA) 
					else 2.0 for atom in mol.GetAtoms()])


def get_atom_coords(mol: Chem.Mol) -> np.ndarray:
	return mol.GetConformer().GetPositions()


class VisibilityChecker:
	"""Check if protein-ligand atom pairs have unobstructed line-of-sight."""

	def __init__(self, protein: Chem.Mol, ligand: Chem.Mol):
		self.prot_coords = get_atom_coords(protein)
		self.lig_coords = get_atom_coords(ligand)
		self.prot_radii = get_vdw_radii(protein)
		self.lig_radii = get_vdw_radii(ligand)
		
		self.all_coords = np.vstack((self.prot_coords, self.lig_coords))
		self.all_radii = np.concatenate((self.prot_radii, self.lig_radii))
		
		self.kdtree = cKDTree(self.all_coords)
		self.max_radius = np.max(self.all_radii)
		self.ligand_offset = len(self.prot_coords)
	
	def batch_check_visibility(self, interactions: List[Tuple[int, int]],
							tol_cutoff: float = -0.5, buffer_dist: float = 0.5) -> List[Tuple[int, int]]:
		"""Filter interactions by ray-casting: a pair is visible if no atom blocks the line between them."""
		if not interactions:
			return []
		
		n_pairs = len(interactions)
		visible = np.ones(n_pairs, dtype=bool)
		
		start_indices = np.zeros(n_pairs, dtype=int)
		end_indices = np.zeros(n_pairs, dtype=int)
		distances = np.zeros(n_pairs)
		
		for i, (prot_idx, lig_idx) in enumerate(interactions):
			start_indices[i] = prot_idx
			end_indices[i] = self.ligand_offset + lig_idx
			distances[i] = np.linalg.norm(self.prot_coords[prot_idx] - self.lig_coords[lig_idx])
		
		for i in range(n_pairs):
			if not visible[i]:
				continue
			
			start = self.all_coords[start_indices[i]]
			end = self.all_coords[end_indices[i]]
			dist = distances[i]
			
			if dist < 1e-10:
				visible[i] = False
				continue
			
			dir_vec = (end - start) / dist

			current_tol = 0.2 if dist > 4.0 else tol_cutoff
			
			blocker_indices = self.kdtree.query_ball_point(start, dist + self.max_radius)
			blocker_indices = [idx for idx in blocker_indices 
							if idx != start_indices[i] and idx != end_indices[i]]
			
			if not blocker_indices:
				continue
			
			blockers = self.all_coords[blocker_indices]
			t = np.dot(blockers - start, dir_vec)

			mask = (t > buffer_dist) & (t < (dist - buffer_dist))
			if not np.any(mask):
				continue
			
			t_valid = t[mask]
			blockers_valid = blockers[mask]
			radii_valid = self.all_radii[np.array(blocker_indices)[mask]]
			
			closest_points = start + t_valid[:, np.newaxis] * dir_vec
			dists_to_line = np.linalg.norm(blockers_valid - closest_points, axis=1)
			
			if np.any(dists_to_line < (radii_valid + current_tol)):
				visible[i] = False
		
		return [interactions[i] for i in range(n_pairs) if visible[i]]

def filter_interactions_los(protein: Chem.Mol, ligand: Chem.Mol,
						interactions: List[Tuple[int, int]],
						tol_cutoff: float = -0.5, buffer_dist: float = 0.5) -> List[Tuple[int, int]]:
	"""Filter protein-ligand atom pairs by line-of-sight: remove pairs blocked by intervening atoms."""
	checker = VisibilityChecker(protein, ligand)
	return checker.batch_check_visibility(interactions, tol_cutoff, buffer_dist)

