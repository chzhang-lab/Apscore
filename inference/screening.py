import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

from model.model import APScore, AtomicPairTransformer, ACANet
from feats.target_opt import opt_target_pdb2pqr
from data.data import ScreeningDataset, collate_fn_infr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pb', '--protein_base', required=True,
                    help='Base directory containing protein folders')
    parser.add_argument('-lb', '--ligand_base', required=True,
                    help='Base directory containing ligand folders')
    parser.add_argument('-m', '--model', required=True,
                    help='Path to trained model')
    parser.add_argument('-o', '--output_dir', default='./screening_results',
                    help='Output directory for saving score files')
    
    parser.add_argument('--atom_feat_dim', type=int, default=23,
                    help='Protein-ligand atom feature dimension')
    parser.add_argument('--bond_feat_dim', type=int, default=6,
                    help='Protein-ligand bond feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='Protein-ligand hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                    help='Protein-ligand number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                    help='Protein-ligand number of transformer layers')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Protein-ligand dropout rate')
    
    return parser.parse_args()


def screen_single_target(protein_file, ligand_dir, model, device, args):
    model.eval()
    torch.set_grad_enabled(False)

    # Check if protein file has already been hydrogenated
    if not protein_file.endswith('_addh.pdb'):
        # Add hydrogens and assign charges to protein
        dir_path = os.path.dirname(protein_file)
        base_name = os.path.basename(protein_file)
        output_addh_name = os.path.splitext(base_name)[0] + "_addh.pdb"
        output_addh_path = os.path.join(dir_path, output_addh_name)
        
        opt_target_pdb2pqr(protein_file)    # Add hydrogens and assign charges using pdb2pqr
        protein_file = output_addh_path 
    
    # Create ScreeningDataset instance
    dataset = ScreeningDataset(
        protein_file=protein_file,
        ligand_dir=ligand_dir,
        parallel=True,
        target_lig_batch=1000,
        max_workers=48
    )
    
    # Get all mol2 files
    mol2_files = dataset.get_mol2_files()
    total_files = len(mol2_files)
    
    # Processing statistics
    stats = {
        'total_files': total_files,
        'processed_files': 0,
        'skipped_files': 0,
        'total_conformers': 0,
        'processed_conformers': 0,
        'skipped_conformers': 0
    }
    
    all_scores = []
    
    for file_idx, mol2_file in enumerate(mol2_files):
        filename = os.path.basename(mol2_file)
        stats['processed_files'] += 1
        
        try:
            print(f"\nProcessing target: {filename}, {file_idx+1}/{total_files}", flush=True)
            
            # Process all conformations in a single mol2 file
            file_conformers_data = dataset.process_single_mol2_file(mol2_file)
            
            if not file_conformers_data:
                print(f"No valid conformations in file {filename}, skipping", flush=True)
                stats['skipped_files'] += 1
                continue
            
            # Get conformer count for statistics
            conformer_count = len(file_conformers_data)
            stats['total_conformers'] += conformer_count
            
            # Create temporary dataset for inference
            class TempDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataloader = DataLoader(
                TempDataset(file_conformers_data),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn_infr
            )
            
            file_scores = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        scores, dist, sigma, _, _ = model(batch)
                        batch_size = scores.shape[0]
                        
                        penalties = calc_clash_penalty(
                            dist,
                            sigma,
                            clash_tol=0.6,
                            penalty_per_pair=0.1
                        )
                        scores = scores - penalties
                        scores = torch.max(scores, torch.zeros_like(scores))

                        for i in range(batch_size):
                            file_scores.append({
                                '#code_ligand_num': batch['pdb_id'][i],
                                'score': scores.cpu().numpy()[i]
                            })
                
                    except Exception as batch_error:
                        # Skip failed conformation but continue with others
                        print(f"Error in conformation {batch_idx+1} of file {filename}: {str(batch_error)}, skipping", flush=True)
                        stats['skipped_conformers'] += 1
                        continue
            
            # Add current file scores to total results
            if file_scores:
                all_scores.extend(file_scores)
                stats['processed_conformers'] += len(file_scores)
                print(f" File {filename} processing completed: {len(file_scores)}/{conformer_count} conformations processed, total scores: {len(all_scores)}", flush=True)
            else:
                stats['skipped_files'] += 1
                print(f"Target {filename} processing completed but no valid results", flush=True)
            
        except Exception as file_error:
            # Skip failed file but continue with others
            print(f"Error processing file {filename}: {str(file_error)}, skipping", flush=True)
            stats['skipped_files'] += 1
            continue

        # Release memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Processing Statistics:")
    print(f"  Files: {stats['processed_files'] - stats['skipped_files']}/{stats['total_files']} files processed successfully")
    print(f"  Conformers: {stats['processed_conformers']}/{stats['total_conformers']} conformers processed successfully")
    print(f"  Skipped: {stats['skipped_files']} files, {stats['skipped_conformers']} conformers")
    print(f"{'='*60}")
    
    return all_scores

def calc_clash_penalty(dist, sigma, clash_tol=0.6, penalty_per_pair=0.1):
    """Calculate clash penalty based on distance and sigma values"""
    clash_pairs = dist < sigma * clash_tol
    clash_counts = clash_pairs.sum(dim=1)
    penalties = clash_counts * penalty_per_pair
    return penalties



def batch_screening(protein_base_dir, ligand_base_dir, model, device, args):
    # Get all PDB subdirectories
    all_dirs = [os.path.join(protein_base_dir, d) for d in os.listdir(protein_base_dir) 
                if os.path.isdir(os.path.join(protein_base_dir, d))]
    
    # Filter directories with valid files
    valid_dirs = []
    for pdb_dir in all_dirs:
        pdb_id = os.path.basename(pdb_dir)
        protein_file = os.path.join(pdb_dir, f"{pdb_id}_pocket_addh.pdb")
        ligand_dir = os.path.join(ligand_base_dir, pdb_id)
        
        if os.path.exists(protein_file) and os.path.exists(ligand_dir):
            valid_dirs.append(pdb_dir)
    print(f"Found {len(valid_dirs)} valid targets", flush=True)

    # Process targets - independent error handling for each target
    for i, pdb_dir in enumerate(valid_dirs):
        pdb_id = os.path.basename(pdb_dir)
        print(f"\n{'='*60}", flush=True)
        print(f"Processing target {i+1}/{len(valid_dirs)}: {pdb_id}", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Find protein file
        protein_file = os.path.join(pdb_dir, f"{pdb_id}_pocket_addh.pdb")
        ligand_dir = os.path.join(ligand_base_dir, pdb_id)
        
        try:
            # Process single target (streaming)
            results = screen_single_target(protein_file, ligand_dir, model, device, args)
            
            if results:
                # Save scores for each target to specified directory
                output_file = os.path.join(args.output_dir, f"{pdb_id}_score.dat")
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_file, index=False, sep='\t')
                print(f"Scores saved to: {output_file}", flush=True)
            else:
                print(f"\nNo valid results for target {pdb_id}", flush=True)
            
        except Exception as e:
            # Skip failed target but continue with others
            print(f"Critical error processing target {pdb_id}: {str(e)}, skipping", flush=True)
            continue
        
        # Clean up memory and temporary files
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\nAll targets processed!", flush=True)
    return None 


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created: {args.output_dir}")

    pl_acanet = ACANet(
        atom_feat_dim=args.atom_feat_dim,
        bond_feat_dim=args.bond_feat_dim,
        hidden_dim=args.hidden_dim,
    )
    aptransformer = AtomicPairTransformer(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    )

    ll_acanet = ACANet(
        atom_feat_dim=args.atom_feat_dim,
        bond_feat_dim=args.bond_feat_dim,
        hidden_dim=args.hidden_dim,
    )
    
    model = APScore(
        pl_acanet,
        aptransformer,
        ll_acanet,
    ).to(device)
    
    # Load trained model
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Perform batch screening
    batch_screening(
        args.protein_base,
        args.ligand_base,
        model,
        device,
        args
    )

if __name__ == '__main__':
    main()
