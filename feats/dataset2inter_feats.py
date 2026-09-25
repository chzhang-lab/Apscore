import os
import gc
import sys
import time
import argparse
import tempfile
import struct

import torch
import numpy as np
from joblib import Parallel, delayed

from mol2inter_feats import cplx_to_interactions				

def read_affinity_data(affinity_file):
	affinity_data = {}
	with open(affinity_file, 'r') as f:
		for line in f:
			if line.startswith('#'): continue
			parts = line.strip().split()
			if len(parts) >= 4:
				affinity_data[parts[0]] = float(parts[3])
	return affinity_data

def process_complex(pdb_id, subdir, affinity_data):
    """Process a single protein-ligand complex"""
    try:
        prot_file = os.path.join(subdir, f"{pdb_id}_pocket_addh.pdb")
        lig_file_mol2 = os.path.join(subdir, f"{pdb_id}_ligand.mol2")
        lig_file_sdf = os.path.join(subdir, f"{pdb_id}_ligand.sdf")
        
        # Check for ligand file
        lig_file = None
        if os.path.exists(lig_file_mol2):
            lig_file = lig_file_mol2
        elif os.path.exists(lig_file_sdf):
            lig_file = lig_file_sdf
        else:
            return None
            
        # Extract complex features
        features = cplx_to_interactions(prot_file, lig_file, is_inference=False)
        if features is None:
            return None
            
        # Unpack features
        pl_dist, prot_atom_feats, lig_atom_feats, prot_neighbor, lig_neighbor, angles, \
        ll_dist, ll_atom1_feats, ll_atom2_feats, ll_neighbor1, ll_neighbor2 = features
        
        # Validate features
        feature_checks = [
            ("pl_dist", pl_dist),
            ("prot_atom_feats", prot_atom_feats),
            ("lig_atom_feats", lig_atom_feats),
            ("prot_neighbor", prot_neighbor),
            ("lig_neighbor", lig_neighbor),
            ("angles", angles),
            ("ll_dist", ll_dist),
            ("ll_atom1_feats", ll_atom1_feats),
            ("ll_atom2_feats", ll_atom2_feats),
            ("ll_neighbor1", ll_neighbor1),
            ("ll_neighbor2", ll_neighbor2)
        ]
        
        for name, feature in feature_checks:
            if feature is None:
                return None

        # Get affinity data if available
        affinity = affinity_data.get(pdb_id, 0.0)
        
        return pdb_id, (pl_dist, ll_dist), \
            (prot_atom_feats, lig_atom_feats, prot_neighbor, lig_neighbor), \
            (ll_atom1_feats, ll_atom2_feats, ll_neighbor1, ll_neighbor2), \
            angles, affinity
            
    except Exception as e:
        return None

def save_results(results, outprefix):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(outprefix), exist_ok=True)
    metadata_file = f"{outprefix}metadata.bin"
    
    # Define output files for each feature
    matrix_files = {
        'pl_dist': f"{outprefix}pl_dist.bin",
        'll_dist': f"{outprefix}ll_dist.bin",
        'prot_atom_feats': f"{outprefix}prot_atom_feats.bin",
        'lig_atom_feats': f"{outprefix}lig_atom_feats.bin",
        'prot_neighbor': f"{outprefix}prot_neighbor.bin",
        'lig_neighbor': f"{outprefix}lig_neighbor.bin",
        'll_atom1_feats': f"{outprefix}ll_atom1_feats.bin",
        'll_atom2_feats': f"{outprefix}ll_atom2_feats.bin",
        'll_neighbor1': f"{outprefix}ll_neighbor1.bin",
        'll_neighbor2': f"{outprefix}ll_neighbor2.bin",
        'angles': f"{outprefix}angles.bin"
    }

    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        return

    # Open all output files
    with open(metadata_file, 'wb') as mf, \
        open(matrix_files['pl_dist'], 'wb') as plf, \
        open(matrix_files['ll_dist'], 'wb') as llf, \
        open(matrix_files['prot_atom_feats'], 'wb') as paf, \
        open(matrix_files['lig_atom_feats'], 'wb') as laf, \
        open(matrix_files['prot_neighbor'], 'wb') as pnf, \
        open(matrix_files['lig_neighbor'], 'wb') as lnf, \
        open(matrix_files['ll_atom1_feats'], 'wb') as la1f, \
        open(matrix_files['ll_atom2_feats'], 'wb') as la2f, \
        open(matrix_files['ll_neighbor1'], 'wb') as ln1f, \
        open(matrix_files['ll_neighbor2'], 'wb') as ln2f, \
        open(matrix_files['angles'], 'wb') as af:  

        # Write number of complexes
        mf.write(struct.pack('I', len(valid_results)))

        # Collect feature dimensions
        pl_dist_lengths, ll_dist_lengths = [], []
        prot_atom_feats_lengths, lig_atom_feats_lengths = [], []
        prot_neighbor_lengths, lig_neighbor_lengths = [], []
        ll_atom1_feats_lengths, ll_atom2_feats_lengths = [], []
        ll_neighbor1_lengths, ll_neighbor2_lengths = [], []
        angles_lengths = []

        processed_count = 0

        for i, result in enumerate(valid_results):
            pdb_id = None
            try:
                pdb_id, matrices, pl_feats, ll_feats, angles, affinity = result
                
                prot_atom_feats, lig_atom_feats, prot_neighbor, lig_neighbor = pl_feats
                ll_atom1_feats, ll_atom2_feats, ll_neighbor1, ll_neighbor2 = ll_feats
                
                # Write basic info
                mf.write(struct.pack('4sf', pdb_id.encode('ascii'), affinity))
                
                # Unpack matrix data
                pl_dist, ll_dist = matrices
                
                # Get feature dimensions
                atom_feat_dim = prot_atom_feats.shape[1]
                neigh_feat_dim = prot_neighbor.shape[2]
                
                # Store feature lengths
                pl_dist_lengths.append(pl_dist.shape[0])
                prot_atom_feats_lengths.append(prot_atom_feats.shape[0] * atom_feat_dim)
                lig_atom_feats_lengths.append(lig_atom_feats.shape[0] * atom_feat_dim)
                prot_neighbor_lengths.append(prot_neighbor.shape[0] * 6 * neigh_feat_dim)
                lig_neighbor_lengths.append(lig_neighbor.shape[0] * 6 * neigh_feat_dim)
                angles_lengths.append(angles.shape[0])

                # Ligand-ligand feature lengths
                ll_dist_lengths.append(ll_dist.shape[0])
                ll_atom1_feats_lengths.append(ll_atom1_feats.shape[0] * atom_feat_dim)
                ll_atom2_feats_lengths.append(ll_atom2_feats.shape[0] * atom_feat_dim)
                ll_neighbor1_lengths.append(ll_neighbor1.shape[0] * 6 * neigh_feat_dim)
                ll_neighbor2_lengths.append(ll_neighbor2.shape[0] * 6 * neigh_feat_dim)

                # Write feature data
                plf.write(pl_dist.cpu().numpy().astype(np.float32).tobytes())
                llf.write(ll_dist.cpu().numpy().astype(np.float32).tobytes())
                
                paf.write(prot_atom_feats.cpu().numpy().astype(np.float32).tobytes())
                laf.write(lig_atom_feats.cpu().numpy().astype(np.float32).tobytes())
                pnf.write(prot_neighbor.cpu().numpy().astype(np.float32).tobytes())
                lnf.write(lig_neighbor.cpu().numpy().astype(np.float32).tobytes())
                af.write(angles.cpu().numpy().astype(np.float32).tobytes())

                la1f.write(ll_atom1_feats.cpu().numpy().astype(np.float32).tobytes())
                la2f.write(ll_atom2_feats.cpu().numpy().astype(np.float32).tobytes())
                ln1f.write(ll_neighbor1.cpu().numpy().astype(np.float32).tobytes())
                ln2f.write(ll_neighbor2.cpu().numpy().astype(np.float32).tobytes())

                processed_count += 1
                
            except Exception as e:
                continue

        # Write feature lengths
        length_lists = [
            pl_dist_lengths,
            prot_atom_feats_lengths, lig_atom_feats_lengths,
            prot_neighbor_lengths, lig_neighbor_lengths, angles_lengths,
            ll_dist_lengths, ll_atom1_feats_lengths, ll_atom2_feats_lengths,
            ll_neighbor1_lengths, ll_neighbor2_lengths
        ]
        
        for i, lengths in enumerate(length_lists):
            if len(lengths) != processed_count:
                # Fill missing lengths with 0
                while len(lengths) < processed_count:
                    lengths.append(0)
            mf.write(struct.pack('I' * len(lengths), *lengths))

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PDBbind dataset into atomic pair features.")

    parser.add_argument("--input_dirs", type=str, nargs="+", required=True,
        help="PDBbind dataset directories containing complex subdirectories")
    parser.add_argument("--output_prefix", type=str, required=True,
        help="Output path prefix for feature files (e.g., ./dataset/pdbbind_dataset/)")
    parser.add_argument("--affinity_file", type=str, required=True,
        help="Path to INDEX_general_PL_data file with affinity data")
    parser.add_argument("--coreset_path", type=str, default=None,
        help="Path to coreset directory for test set exclusion (optional)")
    parser.add_argument("--batch_size", type=int, default=500,
        help="Number of complexes to process per batch")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    # Read affinity data
    print("Reading affinity data...")
    affinity_data = read_affinity_data(args.affinity_file)

    # Get test set PDB IDs
    test_set_pdbs = set()
    if args.coreset_path and os.path.exists(args.coreset_path):
        print("Reading test set PDB IDs...")
        test_set_pdbs = {d for d in os.listdir(args.coreset_path)
                        if os.path.isdir(os.path.join(args.coreset_path, d))}
    else:
        print("Warning: No coreset path provided, all complexes used for training")

    # Get all PDB IDs
    print("Scanning PDB IDs...")
    all_pdb_ids = []

    for main_dir in args.input_dirs:
        if not os.path.exists(main_dir):
            print(f"Warning: Directory not found: {main_dir}")
            continue
        pdb_ids = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
        all_pdb_ids.extend([(pdb_id, main_dir) for pdb_id in pdb_ids])

    # Filter out test set PDB IDs
    original_count = len(all_pdb_ids)
    all_pdb_ids = [(pdb_id, main_dir) for pdb_id, main_dir in all_pdb_ids
                if pdb_id not in test_set_pdbs]
    filtered_count = original_count - len(all_pdb_ids)
    print(f"Found {original_count} PDB IDs total")
    print(f"Excluded {filtered_count} test set PDB IDs")
    print(f"Remaining {len(all_pdb_ids)} PDB IDs for training")

    # Parallel processing
    results = []
    failed_pdbs = []
    print(f"Starting parallel processing with batch size: {args.batch_size}")

    for i in range(0, len(all_pdb_ids), args.batch_size):
        batch_ids = all_pdb_ids[i:i + args.batch_size]
        batch_results = Parallel(n_jobs=-1, prefer="processes", timeout=3600)(
            delayed(process_complex)(pdb_id, os.path.join(main_dir, pdb_id), affinity_data)
            for pdb_id, main_dir in batch_ids)
        failed_pdbs.extend([pdb_id for pdb_id, result in zip([x[0] for x in batch_ids], batch_results) if result is None])

        # Filter valid results
        valid_results = [r for r in batch_results if r is not None]
        results.extend(valid_results)

        # Memory cleanup
        del batch_results, valid_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up temporary files after each batch
        cleanup_temp_files()

    # Save results
    print(f"Saving results to {args.output_prefix}...")
    save_results(results, args.output_prefix)

    total_time = time.time() - start_time
    print("\nProcessing completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Successfully processed: {len(results)} samples")
    print(f"Failed processing: {len(failed_pdbs)} samples")

    if failed_pdbs:
        print(f"Failed PDB IDs: {failed_pdbs}")

    # Final cleanup of temporary files
    cleanup_temp_files()


def cleanup_temp_files(temp_dir=None):
    """Clean up all files in a temporary directory.

    Args:
        temp_dir: Path to the temp directory to clean up.
                  Defaults to system temp dir + '/pdb_surface_temp'.
    """
    if temp_dir is None:
        temp_dir = os.path.join(tempfile.gettempdir(), "pdb_surface_temp")
    try:
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass
            try:
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception:
                pass
    except Exception:
        pass

if __name__ == '__main__':
    main()
