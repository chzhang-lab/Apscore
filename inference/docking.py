import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
import gc
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

from model.model import APScore, AtomicPairTransformer, ACANet
from feats.target_opt import opt_target_pdb2pqr
from data.data import DockingDataset, collate_fn_infr

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-pb', '--protein_base', required=True, 
                help='Base directory containing protein folders')
    parser.add_argument('-lb', '--ligand_base', required=True, 
                help='Base directory containing ligand folders')
    parser.add_argument('-m', '--model', required=True, 
                help='Path to trained model')
    parser.add_argument('-o', '--output_dir', default='./results', 
                help='Output directory')
    
    # Add model parameters
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

# Score docking for a single target
def docking_single_target(protein_file, ligand_dir, model, device, args):
    """
    Streaming scoring for docking of a single target
    """
    model.eval()
    torch.set_grad_enabled(False)

    # Add hydrogens and assign charges to protein
    dir_path = os.path.dirname(protein_file)
    pdb_id = os.path.basename(protein_file).split('_')[0]
    output_addh_name = f"{pdb_id}_pocket_addh.pdb"
    output_addh_path = os.path.join(dir_path, output_addh_name)

    if not os.path.exists(output_addh_path):
        opt_target_pdb2pqr(protein_file)
    
    dataset = DockingDataset(
        protein_file=output_addh_path,
        ligand_dir=ligand_dir,
        parallel=True,
        target_lig_batch=1000,
        max_workers=32
    )
    
    total_files = dataset.get_total_files()
    if total_files == 0:
        return []

    results = []
    batch_count = 0
    
    # Stream process each batch
    for batch_data in dataset.stream_batches():
        batch_count += 1
        
        if not batch_data:
            continue

        class TempDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataloader = DataLoader(
            TempDataset(batch_data),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_infr
        )
        
        # Process current batch
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                score, dist, sigma, _, _ = model(batch)
                batch_size = score.shape[0]
                
                penalties = calc_clash_penalty(
                    dist,
                    sigma, 
                    clash_tol=0.6, 
                    penalty_per_pair=0.0
                )
                score = score - penalties
                score = torch.max(score, torch.zeros_like(score))

                for i in range(batch_size):
                    results.append({
                        '#code': batch['pdb_id'][i],
                        'score': score.cpu().numpy()[i]
                    })
        
        # Release memory for current batch
        del batch_data, dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"  Target {pdb_id} processing completed, total {len(results)} results")
    return results


def calc_clash_penalty(dist, sigma, clash_tol=0.60, penalty_per_pair=0.0):
    """Calculate clash penalty based on distance and sigma values"""
    clash_pairs = dist < sigma * clash_tol
    clash_counts = clash_pairs.sum(dim=1)
    penalties = clash_counts * penalty_per_pair
    return penalties


# Batch docking scoring - sequential processing at target level
def batch_docking(protein_base_dir, ligand_base_dir, model, device, args):
    batch_start_time = time.time()
    
    # Get all pdbids
    pdb_ids = [d for d in os.listdir(protein_base_dir) 
                if os.path.isdir(os.path.join(protein_base_dir, d))]

    print(f"Found {len(pdb_ids)} protein targets to process")
    
    all_results = []
    
    # Sequentially process targets
    for i, pdb_id in enumerate(pdb_ids):
        try:
            protein_file = os.path.join(protein_base_dir, pdb_id, f"{pdb_id}_pocket_addh.pdb")
            ligand_dir = os.path.join(ligand_base_dir, f"{pdb_id}_split_mol")
            
            if not all([os.path.exists(f) for f in [protein_file, ligand_dir]]):
                print(f"Missing files for {pdb_id}, skipping...")
                continue
            
            print(f"\nProcessing target {i+1}/{len(pdb_ids)}: {pdb_id}", flush=True)
            
            # Parallel process ligands within target
            results = docking_single_target(
                protein_file=protein_file,
                ligand_dir=ligand_dir,
                model=model,
                device=device,
                args=args
            )
            
            if results:
                all_results.append(results)
                results_df = pd.DataFrame(results)    # Save results for current target
                results_df = results_df.sort_values('score', ascending=False)
                output_file = os.path.join(args.output_dir, f"{pdb_id}_score.dat")
                results_df.to_csv(output_file, index=False, sep='\t')
                print(f"Results saved to: {output_file}", flush=True)
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {pdb_id}: {str(e)}")
            continue
    
    print(f"\nAll targets processed. Total results: {len(all_results)}")
    batch_end_time = time.time()
    print(f"\nBatch docking completed in {batch_end_time - batch_start_time:.2f} seconds")
    return all_results


def main():
    """Main function for batch docking scoring"""
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Perform batch docking scoring
    all_results = batch_docking(
        args.protein_base,
        args.ligand_base,
        model,
        device,
        args
    )

    # Save results for each target
    for results in all_results:
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('score', ascending=False)  # Sort by score in descending order
            pdb_id = results[0]['#code'][:4] if results else 'unknown'
            output_file = os.path.join(args.output_dir, f"{pdb_id}_score.dat")
            results_df.to_csv(output_file, index=False, sep='\t')
            print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()

