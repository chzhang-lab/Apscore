import torch
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg, SetDihedralDeg

from constants import RCSD_VDW_DATA

class BondSeg():
    def is_carbonyl(self, bond):
        '''
        Check for carbonyl C=O and C=S
        '''
        if bond.GetBondTypeAsDouble() != 2:
            return False

        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 8) or \
        (atom1.GetAtomicNum() == 8 and atom2.GetAtomicNum() == 6):
            return True
        if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 16) or \
        (atom1.GetAtomicNum() == 16 and atom2.GetAtomicNum() == 6):
            return True
        
        return False


    def is_imide(self, bond):
        '''
        Check for C=N
        '''
        if bond.GetBondTypeAsDouble() != 2:
            return False

        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 7) or \
        (atom1.GetAtomicNum() == 7 and atom2.GetAtomicNum() == 6):
            return True

        return False


    def is_amidine(self, bond):
        '''
        Check for amidine, HC(=NH)NH2 and NC=N
        '''
        if bond.GetBondTypeAsDouble() != 1.0:
            return False
        
        c = n = None
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 7):
            c = atom1
            n = atom2
        elif (atom1.GetAtomicNum() == 7 and atom2.GetAtomicNum() == 6):
            c = atom2
            n = atom1

        if not c or not n:
            return False
        
        if n.GetTotalNumHs() + n.GetDegree() != 3:
            return False

        for neighbor_bond in c.GetBonds():
            if self.is_imide(neighbor_bond):
                return True
        return False


    def is_amide(self, bond):
        '''
        Check for amide, HCONH2 and NC=O
        Check for thioamide, NC=S
        '''
        if bond.GetBondTypeAsDouble() != 1.0:
            return False
        
        c = n = None
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 7):
            c = atom1
            n = atom2
        elif (atom1.GetAtomicNum() == 7 and atom2.GetAtomicNum() == 6):
            c = atom2
            n = atom1

        if not c or not n:
            return False

        if n.GetTotalNumHs() + n.GetDegree() != 3:
            return False

        for neighbor_bond in c.GetBonds():
            if self.is_carbonyl(neighbor_bond):
                return True
        return False

    def is_sulfamide(self, bond):
        '''
        Check for sulfamide, O=S(-N*)(*)=O
        '''
        if bond.GetBondTypeAsDouble() != 1.0:
            return False
        
        n = s = None
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 7 and atom2.GetAtomicNum() == 16):
            n = atom1
            s = atom2
        elif (atom1.GetAtomicNum() == 16 and atom2.GetAtomicNum() == 7):
            n = atom2
            s = atom1
        if not n or not s:
            return False
        if n.GetTotalNumHs() + n.GetDegree() != 3:
            return False

        o_num = 0
        for s_nbr_bond in s.GetBonds():
            if s_nbr_bond.GetBondTypeAsDouble() == 2:
                s_nbr_atom = s_nbr_bond.GetOtherAtom(s)
                if s_nbr_atom.GetAtomicNum() == 8:
                    o_num += 1
        if o_num >= 2:
            return True

        return False
    
    def is_rotatable_bond(self, bond):
        # Check if bond type is single bond
        if bond.GetBondTypeAsDouble() != 1.0:
            return False
        # Check if bond is aromatic
        if bond.GetIsAromatic():
            return False
        # Check if bond is (thio)amide or amidine or sulfamide
        if self.is_amide(bond):
            return False
        if self.is_amidine(bond):
            return False
        if self.is_sulfamide(bond):
            return False
        # Check if bond is in ring
        if bond.IsInRing():
            return False
        # Check if bond is terminal bond
        if bond.GetBeginAtom().GetDegree() == 1 or \
        bond.GetEndAtom().GetDegree() == 1:
            return False
        return True
    
    def get_rotate_bonds(self, mol):
        rotatable_bonds = []
        for bond in mol.GetBonds():
            if self.is_rotatable_bond(bond):
                rotatable_bonds.append(bond.GetIdx())
        return rotatable_bonds


def get_rotatable_path_atompairs(mol, rotatable_bonds):
    """Get all atom pairs that include rotatable bonds in their path"""
    rotatable_set = set(rotatable_bonds)
    rotatable_pairs = set()
    num_atoms = mol.GetNumAtoms()
    
    # Use adjacency list for faster path finding
    adj_matrix = {}
    for i in range(num_atoms):
        adj_matrix[i] = set()
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_matrix[i].add(j)
        adj_matrix[j].add(i)
    
    # Generate all possible atom pairs
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            if j not in adj_matrix[i]:  # Not directly connected
                path = Chem.rdmolops.GetShortestPath(mol, i, j)
                rotatable_count = 0
                for k in range(len(path)-1):
                    bond = mol.GetBondBetweenAtoms(path[k], path[k+1])
                    if bond and bond.GetIdx() in rotatable_set:
                        rotatable_count += 1
                        if rotatable_count > 2:
                            break
                if 0 < rotatable_count <= 2:
                    rotatable_pairs.add((i, j) if i < j else (j, i))
    
    return rotatable_pairs


def get_vdw_radii(mol):
    vdw_radii = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        vdw_radius = RCSD_VDW_DATA[atomic_num - 1]
        vdw_radii.append(vdw_radius)
    return vdw_radii


def _get_affected_atoms_by_bond(mol, ba, ea):
    """Get the set of atoms affected by bond rotation (subtree starting from ea)"""
    visited = set([ba])
    stack = [ea]
    affected_atoms = set()
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        
        visited.add(current)
        affected_atoms.add(current)
        
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                stack.append(neighbor_idx)
    
    return affected_atoms


def _rotate_atoms_around_axis(coords, atom_indices, axis_point, axis_direction, angle_deg):
    """Rotate specified atoms around an axis (batch processing)"""
    if len(atom_indices) == 0:
        return coords.copy()
    
    # Create coordinate copy
    rotated_coords = coords.copy()
    
    # Convert angle to radians
    angle_rad = math.radians(angle_deg)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    # Normalize axis direction vector
    axis_norm = np.linalg.norm(axis_direction)
    if axis_norm < 1e-10:
        return rotated_coords
    
    u = axis_direction / axis_norm
    
    # Rodrigues' rotation formula in matrix form
    # R = I + sinθ × K + (1 - cosθ) × K²
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])
    
    K_squared = K @ K
    I = np.eye(3)
    R = I + sin_theta * K + (1 - cos_theta) * K_squared
    
    # Batch rotate atoms
    atoms_to_rotate = np.array(list(atom_indices))
    vectors = coords[atoms_to_rotate] - axis_point
    rotated_vectors = vectors @ R.T  # Note transpose
    rotated_coords[atoms_to_rotate] = axis_point + rotated_vectors
    
    return rotated_coords


'''
Triangle conflict filtering function
'''
def filter_clashes_atom_pairs(mol, coords, clashes_atom_pairs):
    """Filter colliding atom pairs with triangle conflicts"""
    from collections import defaultdict
    
    inv_neighbors = defaultdict(set)
    for a, b in clashes_atom_pairs:
        inv_neighbors[a].add(b)
        inv_neighbors[b].add(a)

    to_remove = set()
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        common = inv_neighbors[u].intersection(inv_neighbors[v])
        for t in common:
            du = np.linalg.norm(coords[t] - coords[u])
            dv = np.linalg.norm(coords[t] - coords[v])
            if du < dv:
                to_remove.add(tuple(sorted((t, v))))
            elif dv < du:
                to_remove.add(tuple(sorted((t, u))))
            else:
                pass

    filtered_pair_set = set(clashes_atom_pairs) - to_remove
    filtered_pairs = list(filtered_pair_set)
    return filtered_pairs


def extract_atom_pairs(mol, step_deg=15, clash_threshold=1.1):
    """
    Optimized small molecule atom pair extraction
    Maintains original computational logic and results while significantly improving performance
    """
    conf = mol.GetConformer()
    bond_seg = BondSeg()
    rotatable_bonds = bond_seg.get_rotate_bonds(mol)
    
    if not rotatable_bonds:
        return [], []
    
    # 1. Precompute all atom coordinates
    num_atoms = mol.GetNumAtoms()
    original_coords = np.zeros((num_atoms, 3))
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        original_coords[i] = [pos.x, pos.y, pos.z]
    
    # 2. Get rotatable path atom pairs
    rotatable_pairs = get_rotatable_path_atompairs(mol, rotatable_bonds)
    
    if not rotatable_pairs:
        return [], []
    
    # Convert to numpy array for vectorized operations
    pairs_list = list(rotatable_pairs)
    pairs_array = np.array(pairs_list)
    
    # 3. Precompute van der Waals radius thresholds
    vdw_radii = get_vdw_radii(mol)
    thresholds = np.array([(vdw_radii[i] + vdw_radii[j]) * clash_threshold 
                        for i, j in pairs_list])
    
    # 4. Precompute initial distances
    initial_distances = np.linalg.norm(original_coords[pairs_array[:, 0]] - 
                                    original_coords[pairs_array[:, 1]], axis=1)
    
    # 5. Distance pruning: only check potentially colliding atom pairs
    max_threshold = np.max(thresholds) if len(thresholds) > 0 else 0
    safe_factor = 3.0  # Safety factor to ensure no possible collision is missed
    distance_mask = initial_distances < (max_threshold * safe_factor)
    
    filtered_pairs = pairs_array[distance_mask]
    filtered_thresholds = thresholds[distance_mask]
    filtered_initial_dists = initial_distances[distance_mask]
    
    if len(filtered_pairs) == 0:
        return [], []
    
    clashes_atom_pairs = set()
    changed_pairs = set()
    
    # 6. Process each rotatable bond
    for bond_idx in rotatable_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        ba = bond.GetBeginAtomIdx()
        ea = bond.GetEndAtomIdx()

        neighbor1 = [n.GetIdx() for n in mol.GetAtomWithIdx(ba).GetNeighbors() if n.GetIdx() != ea]
        neighbor2 = [n.GetIdx() for n in mol.GetAtomWithIdx(ea).GetNeighbors() if n.GetIdx() != ba]
        
        if not neighbor1 or not neighbor2:
            continue
        
        idx1 = neighbor1[0]
        idx4 = neighbor2[0]
        original_dihedral = GetDihedralDeg(conf, idx1, ba, ea, idx4)
        
        # 7. Identify atoms affected by this bond rotation
        affected_atoms = _get_affected_atoms_by_bond(mol, ba, ea)
        
        # 8. Only check atom pairs involving affected atoms
        affected_mask = np.zeros(len(filtered_pairs), dtype=bool)
        for idx, (i, j) in enumerate(filtered_pairs):
            if i in affected_atoms or j in affected_atoms:
                affected_mask[idx] = True
        
        if not np.any(affected_mask):
            continue
        
        affected_pairs = filtered_pairs[affected_mask]
        affected_thresholds = filtered_thresholds[affected_mask]
        affected_initial_dists = filtered_initial_dists[affected_mask]
        
        # 9. Precompute rotation axis
        axis_start = original_coords[ba]
        axis_end = original_coords[ea]
        axis_vector = axis_end - axis_start
        axis_norm = np.linalg.norm(axis_vector)
        
        if axis_norm < 1e-10:
            continue
        
        axis_vector = axis_vector / axis_norm
        
        # 10. Rotate and check for collisions
        prev_distances = affected_initial_dists.copy()
        
        for angle in range(0, 360, step_deg):
            # 11. Calculate rotated coordinates (avoid calling SetDihedralDeg)
            rotated_coords = _rotate_atoms_around_axis(
                original_coords, affected_atoms, axis_start, axis_vector, angle
            )
            
            # 12. Calculate distances for affected atom pairs (vectorized)
            curr_distances = np.linalg.norm(
                rotated_coords[affected_pairs[:, 0]] - rotated_coords[affected_pairs[:, 1]], 
                axis=1
            )
            
            # 13. Check distance changes
            changed_mask = ~np.isclose(curr_distances, prev_distances, rtol=1e-5)
            changed_indices = np.where(changed_mask)[0]
            
            for idx in changed_indices:
                i, j = affected_pairs[idx]
                changed_pairs.add((i, j))
            
            # 14. Check for collisions
            clash_mask = curr_distances < affected_thresholds
            clash_indices = np.where(clash_mask)[0]
            
            for idx in clash_indices:
                i, j = affected_pairs[idx]
                # Ensure consistent atom pair ordering
                if i < j:
                    clashes_atom_pairs.add((i, j))
                else:
                    clashes_atom_pairs.add((j, i))
            
            prev_distances = curr_distances
        
        # 15. Reset dihedral angle
        SetDihedralDeg(conf, idx1, ba, ea, idx4, original_dihedral)
    
    # 16. Post-processing: take intersection
    clashes_atom_pairs = clashes_atom_pairs & changed_pairs
    clashes_atom_pairs = sorted(clashes_atom_pairs)
    
    # 17. Apply triangle conflict filtering
    if clashes_atom_pairs:
        clashes_atom_pairs = filter_clashes_atom_pairs(mol, original_coords, clashes_atom_pairs)
    
    # 18. Calculate final distances
    distances = []
    for i, j in clashes_atom_pairs:
        dist = np.linalg.norm(original_coords[i] - original_coords[j])
        distances.append([float(format(dist, ".2f"))])
    
    return list(clashes_atom_pairs), torch.tensor(np.array(distances))
