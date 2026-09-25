import os
import sys
import uuid
import tempfile
import struct

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
feats_dir = os.path.join(project_root, 'feats')
sys.path.append(feats_dir)

from mol2inter_feats import cplx_to_interactions

class ScoringDataset(Dataset):
    """
    Dataset class for protein-ligand scoring power.
    """
    def __init__(self, protein_file, ligand_files, parallel=False):
        self.protein_file = protein_file
        self.ligand_files = ligand_files
        self.parallel = parallel
        self.data = []
        
        # Process ligands in parallel or sequentially
        if parallel:
            self.data = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._process_complex)(lig_file) for lig_file in ligand_files
            )
        else:
            for lig_file in ligand_files:
                self.data.append(self._process_complex(lig_file))
                
        # Filter out None values
        self.data = list(filter(lambda x: x is not None, self.data))
        
    def _process_complex(self, lig_file):
        try:
            # Get PDB ID from ligand filename
            pdb_id = os.path.basename(lig_file).split('.')[0]
            
            # Extract protein-ligand complex features
            pl_dist, prot_atom_feats, lig_atom_feats, prot_neighbor, lig_neighbor, angles, \
            ll_dist, ll_atom1_feats, ll_atom2_feats, ll_neighbor1, ll_neighbor2 = cplx_to_interactions(
                self.protein_file, lig_file, is_inference=True
            )
            
            return {
                'pdb_id': pdb_id,
                'pl_dist': pl_dist,
                'll_dist': ll_dist,
                
                # Protein-ligand features
                'prot_atom_feats': prot_atom_feats,
                'lig_atom_feats': lig_atom_feats,
                'prot_neighbor': prot_neighbor,
                'lig_neighbor': lig_neighbor,
                
                # Ligand-ligand features
                'll_atom1_feats': ll_atom1_feats,
                'll_atom2_feats': ll_atom2_feats,
                'll_neighbor1': ll_neighbor1,
                'll_neighbor2': ll_neighbor2,
                
                # Angles
                'angles': angles,
            }
        except Exception as e:
            print(f"Failed to process complex with ligand {lig_file}: {str(e)}")
            return None

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        return self.data[idx]




class DockingDataset:
    """
    Dataset class for protein-ligand docking power.
    """
    def __init__(self, protein_file, ligand_dir, parallel=True, target_lig_batch=1000, max_workers=40):
        self.protein_file = protein_file
        self.ligand_dir = ligand_dir
        self.parallel = parallel
        self.target_lig_batch = target_lig_batch
        self.max_workers = max_workers
        
        # Get all ligand files
        self.ligand_files = [os.path.join(ligand_dir, f) for f in os.listdir(ligand_dir) 
                    if f.endswith('.mol2') or f.endswith('.sdf')]
        self.total_files = len(self.ligand_files)
    
    def get_total_files(self):
        """Get total number of ligand files."""
        return self.total_files
    
    def get_ligand_files(self):
        """Get list of all ligand files."""
        return self.ligand_files
    
    def process_single_ligand(self, ligand_file):
        try:
            # Extract features using existing logic
            pl_dist, prot_atom_feats, lig_atom_feats, prot_neighbor, lig_neighbor, angles, \
            ll_dist, ll_atom1_feats, ll_atom2_feats, ll_neighbor1, ll_neighbor2 = cplx_to_interactions(
                self.protein_file, ligand_file, is_inference=True
            )
            
            return {
                'pdb_id': os.path.basename(ligand_file).split('.')[0],
                'pl_dist': pl_dist,
                'll_dist': ll_dist,
                # Protein-ligand features
                'prot_atom_feats': prot_atom_feats,
                'lig_atom_feats': lig_atom_feats,
                'prot_neighbor': prot_neighbor,
                'lig_neighbor': lig_neighbor,
                # Ligand-ligand features
                'll_atom1_feats': ll_atom1_feats,
                'll_atom2_feats': ll_atom2_feats,
                'll_neighbor1': ll_neighbor1,
                'll_neighbor2': ll_neighbor2,
                # Angles
                'angles': angles,
            }
            
        except Exception as e:
            print(f"Failed to process ligand {os.path.basename(ligand_file)}: {str(e)}", flush=True)
            return None
    
    def process_batch(self, batch_start, batch_end=None):
        if batch_end is None:
            batch_end = min(batch_start + self.target_lig_batch, self.total_files)
        
        batch_files = self.ligand_files[batch_start:batch_end]
        
        # Process current batch in parallel or sequentially
        if self.parallel and len(batch_files) > 0:
            n_jobs = min(self.max_workers, len(batch_files))
            batch_data = Parallel(n_jobs=n_jobs)(
                delayed(self.process_single_ligand)(lig_file) for lig_file in batch_files
            )
        else:
            batch_data = []
            for lig_file in batch_files:
                batch_data.append(self.process_single_ligand(lig_file))
        
        # Filter out failed processing
        valid_data = list(filter(lambda x: x is not None, batch_data))
        print(f"Batch complete: {len(valid_data)}/{len(batch_files)} ligands processed successfully", flush=True)
        
        return valid_data
    
    def stream_batches(self):
        for batch_start in range(0, self.total_files, self.target_lig_batch):
            batch_data = self.process_batch(batch_start)
            if batch_data:
                yield batch_data



class ScreeningDataset(Dataset):
    """
    Dataset class for virtual screening power.
    """
    
    def __init__(self, protein_file, ligand_dir, parallel=True, target_lig_batch=1000, max_workers=40):
        self.protein_file = protein_file
        self.ligand_dir = ligand_dir
        self.parallel = parallel
        self.target_lig_batch = target_lig_batch
        self.max_workers = max_workers
        self.data = []
        
        # Stream processing
        self.ligand_files = [os.path.join(ligand_dir, f) for f in os.listdir(ligand_dir) 
                            if f.endswith('.mol2')]
        self.total_files = len(self.ligand_files)
        print(f"Found {self.total_files} mol2 files in directory {ligand_dir}", flush=True)
    
    def process_single_mol2_file(self, ligand_file):
        """
        Process a single mol2 file and return features for all conformers.
        
        Args:
            ligand_file: Path to the mol2 file
            
        Returns:
            list: List of processed conformer data
        """
        filename = os.path.basename(ligand_file)
        try:
            # Split mol2 file into conformers
            conformers = self._mol2_split(ligand_file)
            conformer_count = len(conformers)
            
            file_conformers = []
            for i, conformer_content in enumerate(conformers):
                ligand_name = self._extract_ligand_name(conformer_content)
                if ligand_name:
                    file_conformers.append({
                        'conformer_content': conformer_content,
                        'conformer_idx': i,
                        'ligand_name': ligand_name,
                        'source_file': filename
                    })
            
            # Process conformers in parallel or sequentially
            if self.parallel and len(file_conformers) > 0:
                n_jobs = min(self.max_workers, len(file_conformers))
                batch_data = Parallel(n_jobs=n_jobs)(
                    delayed(self._process_single_conformer)(conformer_data) 
                    for conformer_data in file_conformers
                )
            else:
                batch_data = []
                for conformer_data in file_conformers:
                    batch_data.append(self._process_single_conformer(conformer_data))
            
            # Filter valid data
            valid_data = list(filter(lambda x: x is not None, batch_data))
            print(f"Processed {filename}: {len(valid_data)}/{conformer_count} conformers", flush=True)
            
            return valid_data
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}", flush=True)
            return []
    
    def get_mol2_files(self):
        """Get list of all mol2 files."""
        return self.ligand_files
    
    def _mol2_split(self, infile):
        """Split mol2 file in memory without temporary files."""
        try:
            with open(infile, 'r', buffering=8192) as f:
                contents = f.read()
            return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
        except Exception as e:
            print(f"Error reading file {infile}: {str(e)}", flush=True)
            return []
    
    def _extract_ligand_name(self, conformer_content):
        """Extract ligand name from conformer content."""
        lines = conformer_content.strip().split('\n')
        for line in lines:
            if line.startswith('@<TRIPOS>MOLECULE'):
                continue
            if line.strip() and not line.startswith('#'):
                return line.strip()
        return None
    
    def _process_single_conformer(self, conformer_data):
        """
        Process a single conformer.
        
        Args:
            conformer_data: Dictionary containing conformer data
            
        Returns:
            dict: Dictionary containing processed features or None if processing failed
        """
        temp_file_path = None
        try:
            conformer_content = conformer_data['conformer_content']
            ligand_name = conformer_data['ligand_name']
            
            temp_file_path = create_temp_file(suffix='.mol2', prefix='temp_')
            # Write to temporary file
            with open(temp_file_path, 'w') as temp_file:
                temp_file.write(conformer_content)

            # Extract interaction features
            pl_dist, prot_atom_feats, lig_atom_feats, prot_neighbor, lig_neighbor, angles, \
            ll_dist, ll_atom1_feats, ll_atom2_feats, ll_neighbor1, ll_neighbor2 = cplx_to_interactions(
                self.protein_file, temp_file_path, is_inference=True
            )

            ligand_code = ligand_name
            return {
                'pdb_id': ligand_code,
                'pl_dist': pl_dist,
                'll_dist': ll_dist,
                # Protein-ligand features
                'prot_atom_feats': prot_atom_feats,
                'lig_atom_feats': lig_atom_feats,
                'prot_neighbor': prot_neighbor,
                'lig_neighbor': lig_neighbor,
                # Ligand-ligand features
                'll_atom1_feats': ll_atom1_feats,
                'll_atom2_feats': ll_atom2_feats,
                'll_neighbor1': ll_neighbor1,
                'll_neighbor2': ll_neighbor2,
                # Angles
                'angles': angles,
            }
            
        except Exception as feature_error:
            print(f"Feature extraction failed for {conformer_data.get('ligand_name', 'unknown')}: {feature_error}", flush=True)
            return None
            
        finally:
            # Ensure cleanup of temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                cleanup_temp_file(temp_file_path)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        return self.data[idx]


def collate_fn_infr(batch):
    """
    Collate function for inference batch processing.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        dict: Dictionary containing batched features
    """
    return {
        'pdb_id': [item['pdb_id'] for item in batch],
        'pl_dist': [item['pl_dist'] for item in batch],
        'll_dist': [item['ll_dist'] for item in batch],
        # Protein-ligand features
        'prot_atom_feats': [item['prot_atom_feats'] for item in batch],
        'lig_atom_feats': [item['lig_atom_feats'] for item in batch],
        'prot_neighbor': [item['prot_neighbor'] for item in batch],
        'lig_neighbor': [item['lig_neighbor'] for item in batch],
        # Ligand-ligand features
        'll_atom1_feats': [item['ll_atom1_feats'] for item in batch],
        'll_atom2_feats': [item['ll_atom2_feats'] for item in batch],
        'll_neighbor1': [item['ll_neighbor1'] for item in batch],
        'll_neighbor2': [item['ll_neighbor2'] for item in batch],
        # Angle features
        'angles': [item['angles'] for item in batch],
    }




class PDBbindDataset(Dataset):
    """
    Dataset class for PDBbind v2020 protein-ligand complexes.
    """
    
    def __init__(self, metadata_file,
                 pl_dist_file, ll_dist_file,
                 prot_atom_feats_file, lig_atom_feats_file,
                 prot_neighbor_file, lig_neighbor_file, angles_file,
                 ll_atom1_feats_file, ll_atom2_feats_file,
                 ll_neighbor1_file, ll_neighbor2_file):
        
        # Base file paths
        self.metadata_file = metadata_file
        
        # Protein-ligand feature files
        self.pl_dist_file = pl_dist_file
        self.prot_atom_feats_file = prot_atom_feats_file
        self.lig_atom_feats_file = lig_atom_feats_file
        self.prot_neighbor_file = prot_neighbor_file
        self.lig_neighbor_file = lig_neighbor_file
        self.angles_file = angles_file
        
        # Ligand-ligand feature files
        self.ll_dist_file = ll_dist_file
        self.ll_atom1_feats_file = ll_atom1_feats_file
        self.ll_atom2_feats_file = ll_atom2_feats_file
        self.ll_neighbor1_file = ll_neighbor1_file
        self.ll_neighbor2_file = ll_neighbor2_file
        
        # Load metadata and length information
        self.pdb_ids, self.affinities, self.lengths = self._load_metadata()
        self.atom_pairs_count = self.lengths['pl_dist']

        # Load data using memory mapping
        self._load_memmap_data()
        
        # Auto-detect feature dimensions
        self.atom_feature_dim = self._detect_atom_feature_dimension()
        self.neighbor_feature_dimension = self._detect_neighbor_feature_dimension()
        
        print(f"Dataset loaded: {len(self.pdb_ids)} complexes")
        print(f"Auto-detected feature dimensions - Atom: {self.atom_feature_dim}, Neighbor: {self.neighbor_feature_dimension}")

    def _detect_atom_feature_dimension(self):
        # Detect actual dimension from first valid sample
        for i in range(len(self.pdb_ids)):
            pl_pairs_count = self.lengths['pl_dist'][i]
            if pl_pairs_count > 0:
                start_idx = sum(self.lengths['prot_atom_feats'][:i])
                end_idx = start_idx + self.lengths['prot_atom_feats'][i]
                if end_idx > start_idx:
                    atom_feat_dim = (end_idx - start_idx) // pl_pairs_count
                    if atom_feat_dim > 0:
                        return atom_feat_dim
        return None

    def _detect_neighbor_feature_dimension(self):
        # Detect actual dimension from first valid sample
        for i in range(len(self.pdb_ids)):
            pl_pairs_count = self.lengths['pl_dist'][i]
            if pl_pairs_count > 0:
                start_idx = sum(self.lengths['prot_neighbor'][:i])
                end_idx = start_idx + self.lengths['prot_neighbor'][i]
                if end_idx > start_idx:
                    neighbor_feat_dim = (end_idx - start_idx) // (pl_pairs_count * 6)
                    if neighbor_feat_dim > 0:
                        return neighbor_feat_dim
        
        return None

    def _load_metadata(self):
        pdb_ids = []
        affinities = []
        
        length_keys_order = [
            'pl_dist', 
            'prot_atom_feats', 'lig_atom_feats',
            'prot_neighbor', 'lig_neighbor', 'angles',
            'll_dist', 'll_atom1_feats', 'll_atom2_feats',
            'll_neighbor1', 'll_neighbor2'
        ]
        
        lengths = {key: [] for key in length_keys_order}
        
        with open(self.metadata_file, 'rb') as f:
            # Read number of complexes
            num_complexes_data = f.read(4)
            if len(num_complexes_data) < 4:
                raise ValueError("Incomplete metadata file, cannot read number of complexes")
            
            num_complexes = struct.unpack('I', num_complexes_data)[0]
            print(f"Number of complexes in metadata: {num_complexes}")
            
            # Read PDB IDs and affinities
            for i in range(num_complexes):
                pdb_id_data = f.read(4)
                if len(pdb_id_data) < 4:
                    raise ValueError(f"Complex {i+1}: Cannot read PDB ID, file may be corrupted")
                
                affinity_data = f.read(4)
                if len(affinity_data) < 4:
                    raise ValueError(f"Complex {i+1}: Cannot read affinity, file may be corrupted")
                
                pdb_id = struct.unpack('4s', pdb_id_data)[0].decode('ascii')
                affinity = struct.unpack('f', affinity_data)[0]
                pdb_ids.append(pdb_id)
                affinities.append(affinity)
            
            # Read feature lengths in order
            for key in length_keys_order:
                data = f.read(4 * num_complexes)
                if len(data) != 4 * num_complexes:
                    raise ValueError(f"Incomplete {key} length data, expected {4 * num_complexes} bytes, got {len(data)} bytes")
                lengths[key] = list(struct.unpack('I' * num_complexes, data))
        
        return pdb_ids, affinities, lengths

    def _load_memmap_data(self):
        # Protein-ligand features
        self.pl_dist_data = np.memmap(self.pl_dist_file, dtype='float32', mode='r')
        self.prot_atom_feats_data = np.memmap(self.prot_atom_feats_file, dtype='float32', mode='r')
        self.lig_atom_feats_data = np.memmap(self.lig_atom_feats_file, dtype='float32', mode='r')
        self.prot_neighbor_data = np.memmap(self.prot_neighbor_file, dtype='float32', mode='r')
        self.lig_neighbor_data = np.memmap(self.lig_neighbor_file, dtype='float32', mode='r')
        self.angles_data = np.memmap(self.angles_file, dtype='float32', mode='r')
        
        # Ligand-ligand features
        self.ll_dist_data = np.memmap(self.ll_dist_file, dtype='float32', mode='r')
        self.ll_atom1_feats_data = np.memmap(self.ll_atom1_feats_file, dtype='float32', mode='r')
        self.ll_atom2_feats_data = np.memmap(self.ll_atom2_feats_file, dtype='float32', mode='r')
        self.ll_neighbor1_data = np.memmap(self.ll_neighbor1_file, dtype='float32', mode='r')
        self.ll_neighbor2_data = np.memmap(self.ll_neighbor2_file, dtype='float32', mode='r')

        self.bins = pd.qcut(self.atom_pairs_count, q=16, labels=False, duplicates='drop')

    def __len__(self):
        """Return the number of complexes in the dataset."""
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        """
        Get all features for a single complex.
        """
        pdb_id = self.pdb_ids[idx]
        
        # Calculate start and end indices for each feature
        start_indices = {key: sum(self.lengths[key][:idx]) for key in self.lengths}
        end_indices = {key: start_indices[key] + self.lengths[key][idx] for key in self.lengths}
        
        # Calculate number of interaction pairs
        pl_pairs_count = end_indices['pl_dist'] - start_indices['pl_dist']
        ll_pairs_count = end_indices['ll_dist'] - start_indices['ll_dist']
        
        try:
            # Extract protein-ligand features
            pl_features = self._extract_pl_features(start_indices, end_indices, pl_pairs_count)
            
            # Extract ligand-ligand features
            ll_features = self._extract_ll_features(start_indices, end_indices, ll_pairs_count)
            
            # Return assembled data
            return {
                'pdb_id': pdb_id,
                
                # Protein-ligand features
                'pl_dist': pl_features['pl_dist'],
                'prot_atom_feats': pl_features['prot_atom_feats'],
                'lig_atom_feats': pl_features['lig_atom_feats'],
                'prot_neighbor': pl_features['prot_neighbor'],
                'lig_neighbor': pl_features['lig_neighbor'],
                'angles': pl_features['angles'],
                
                # Ligand-ligand features
                'll_dist': ll_features['ll_dist'],
                'll_atom1_feats': ll_features['ll_atom1_feats'],
                'll_atom2_feats': ll_features['ll_atom2_feats'],
                'll_neighbor1': ll_features['ll_neighbor1'],
                'll_neighbor2': ll_features['ll_neighbor2'],
                
                # Labels
                'affinity': torch.tensor(self.affinities[idx], dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"Error loading sample {idx} (PDB: {pdb_id}): {e}")
            print(f"PL interaction pairs: {pl_pairs_count}, LL interaction pairs: {ll_pairs_count}")
            raise e

    def _extract_pl_features(self, start_indices, end_indices, pl_pairs_count):
        if pl_pairs_count == 0:
            return {
                'pl_dist': torch.empty(0, 1),
                'prot_atom_feats': torch.empty(0, self.atom_feature_dim),
                'lig_atom_feats': torch.empty(0, self.atom_feature_dim),
                'prot_neighbor': torch.empty(0, 6, self.neighbor_feature_dimension),
                'lig_neighbor': torch.empty(0, 6, self.neighbor_feature_dimension),
                'angles': torch.empty(0, 1)
            }
        
        # Distance features
        pl_dist = torch.from_numpy(
            self.pl_dist_data[start_indices['pl_dist']:end_indices['pl_dist']].copy()
        ).reshape(-1, 1)
        
        # Atom features
        prot_atom_feats = torch.from_numpy(
            self.prot_atom_feats_data[start_indices['prot_atom_feats']:end_indices['prot_atom_feats']].copy()
        ).reshape(-1, self.atom_feature_dim)
        
        lig_atom_feats = torch.from_numpy(
            self.lig_atom_feats_data[start_indices['lig_atom_feats']:end_indices['lig_atom_feats']].copy()
        ).reshape(-1, self.atom_feature_dim)
        
        # Neighbor features
        prot_neighbor = torch.from_numpy(
            self.prot_neighbor_data[start_indices['prot_neighbor']:end_indices['prot_neighbor']].copy()
        ).reshape(-1, 6, self.neighbor_feature_dimension)
        
        lig_neighbor = torch.from_numpy(
            self.lig_neighbor_data[start_indices['lig_neighbor']:end_indices['lig_neighbor']].copy()
        ).reshape(-1, 6, self.neighbor_feature_dimension)
        
        # Angle features
        angles = torch.from_numpy(
            self.angles_data[start_indices['angles']:end_indices['angles']].copy()
        ).reshape(-1, 1)
        
        return {
            'pl_dist': pl_dist,
            'prot_atom_feats': prot_atom_feats,
            'lig_atom_feats': lig_atom_feats,
            'prot_neighbor': prot_neighbor,
            'lig_neighbor': lig_neighbor,
            'angles': angles
        }

    def _extract_ll_features(self, start_indices, end_indices, ll_pairs_count):
        if ll_pairs_count == 0:
            return {
                'll_dist': torch.empty(0, 1),
                'll_atom1_feats': torch.empty(0, self.atom_feature_dim),
                'll_atom2_feats': torch.empty(0, self.atom_feature_dim),
                'll_neighbor1': torch.empty(0, 6, self.neighbor_feature_dimension),
                'll_neighbor2': torch.empty(0, 6, self.neighbor_feature_dimension)
            }
        
        # Distance features
        ll_dist = torch.from_numpy(
            self.ll_dist_data[start_indices['ll_dist']:end_indices['ll_dist']].copy()
        ).reshape(-1, 1)
        
        # Atom features
        ll_atom1_feats = torch.from_numpy(
            self.ll_atom1_feats_data[start_indices['ll_atom1_feats']:end_indices['ll_atom1_feats']].copy()
        ).reshape(-1, self.atom_feature_dim)
        
        ll_atom2_feats = torch.from_numpy(
            self.ll_atom2_feats_data[start_indices['ll_atom2_feats']:end_indices['ll_atom2_feats']].copy()
        ).reshape(-1, self.atom_feature_dim)
        
        # Neighbor features
        ll_neighbor1 = torch.from_numpy(
            self.ll_neighbor1_data[start_indices['ll_neighbor1']:end_indices['ll_neighbor1']].copy()
        ).reshape(-1, 6, self.neighbor_feature_dimension)
        
        ll_neighbor2 = torch.from_numpy(
            self.ll_neighbor2_data[start_indices['ll_neighbor2']:end_indices['ll_neighbor2']].copy()
        ).reshape(-1, 6, self.neighbor_feature_dimension)
        
        return {
            'll_dist': ll_dist,
            'll_atom1_feats': ll_atom1_feats,
            'll_atom2_feats': ll_atom2_feats,
            'll_neighbor1': ll_neighbor1,
            'll_neighbor2': ll_neighbor2
        }

    def get_stratification_labels(self):
        return self.bins


def collate_fn(batch):
    """
    Custom collate function for training batch processing.
    """
    return {
        'pdb_id': [item['pdb_id'] for item in batch],
        'pl_dist': [item['pl_dist'] for item in batch],
        'prot_atom_feats': [item['prot_atom_feats'] for item in batch],
        'lig_atom_feats': [item['lig_atom_feats'] for item in batch],
        'prot_neighbor': [item['prot_neighbor'] for item in batch],
        'lig_neighbor': [item['lig_neighbor'] for item in batch],
        'angles': [item['angles'] for item in batch],
        'll_dist': [item['ll_dist'] for item in batch],
        'll_atom1_feats': [item['ll_atom1_feats'] for item in batch],
        'll_atom2_feats': [item['ll_atom2_feats'] for item in batch],
        'll_neighbor1': [item['ll_neighbor1'] for item in batch],
        'll_neighbor2': [item['ll_neighbor2'] for item in batch],
        'affinity': torch.stack([item['affinity'] for item in batch])
    }


def get_stratified_split(dataset, val_split=0.2, seed=42):
    # Get stratification labels
    labels = dataset.get_stratification_labels()
    
    # Create stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    return train_dataset, val_dataset



def create_temp_file(suffix=".tmp", prefix="temp_"):
    """Create a temporary file path in the system temp directory."""
    tmp_dir = tempfile.gettempdir()
    temp_filename = f"{prefix}{uuid.uuid4().hex[:8]}{suffix}"
    temp_filepath = os.path.join(tmp_dir, temp_filename)
    return temp_filepath



def cleanup_temp_file(temp_filepath):
    if os.path.exists(temp_filepath):
        try:
            os.remove(temp_filepath)
        except:
            pass