import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import argparse

from model.model import APScore, AtomicPairTransformer, ACANet
from feats.target_opt import opt_target_pdb2pqr
from data.data import ScoringDataset, collate_fn_infr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], required=True, 
            help='Scoring mode: single complex or batch complexes')
    parser.add_argument('-p', '--protein_file', 
            help='Input protein file for single mode')
    parser.add_argument('-l', '--ligand_file', 
            help='Input ligand file for single mode')
    parser.add_argument('--input_dir', 
            help='Input directory containing complex folders for batch mode')
    parser.add_argument('-m', '--model', required=True, 
            help='Trained model path')
    parser.add_argument('-o', '--output', default='scoring_results.dat', 
            help='Output file name')
    
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


# Score a single complex
def scoring_single(protein_file, ligand_file, model, device):
    """Score a single protein-ligand complex"""
    model.eval()
    torch.set_grad_enabled(False)    # Disable gradients

    # Add hydrogens and assign charges to protein
    dir_path = os.path.dirname(protein_file)
    base_name = os.path.basename(protein_file)
    output_addh_name = os.path.splitext(base_name)[0] + "_addh.pdb"
    output_addh_path = os.path.join(dir_path, output_addh_name)
    
    opt_target_pdb2pqr(protein_file)    # Add hydrogens using pdb2pqr
    
    # Create scoring dataset
    dataset = ScoringDataset(
        protein_file=output_addh_path,
        ligand_files=[ligand_file],
        parallel=True
    )

    # Single complex data loader
    single_loader = DataLoader(
        dataset,
        batch_size=1,                # CASF scoring test uses bs=1
        shuffle=False,              
        num_workers=0, 
        collate_fn=collate_fn_infr
    )

    with torch.no_grad():
        for batch in single_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get score and sigma
            score, Dist, sigma, pl_probs, w = model(batch)
            
            # Apply clash penalty
            penalties = calc_clash_penalty(Dist, sigma, clash_tol=0.6, penalty_per_pair=0.1)
            score = score - penalties
            score = torch.max(score, torch.zeros_like(score))
            
            return batch['pdb_id'], score.cpu().numpy(), sigma.cpu().numpy(), pl_probs.cpu().numpy(), w.cpu().numpy()

def calc_clash_penalty(dist, sigma, clash_tol=0.6, penalty_per_pair=0.1):
    """Calculate clash penalty based on distance and sigma values"""
    clash_pairs = dist < sigma * clash_tol
    clash_counts = clash_pairs.sum(dim=1)
    penalties = clash_counts * penalty_per_pair
    return penalties


# Batch scoring of multiple complexes
def batch_scoring(input_dir, model, device):
    # Get all PDB subdirectories
    pdb_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) 
                if os.path.isdir(os.path.join(input_dir, d))]

    # Define single complex processing function
    def process_single_complex(pdb_dir):
        try:
            pdb_id = os.path.basename(pdb_dir)
            protein_file = os.path.join(pdb_dir, f"{pdb_id}_pocket_addh.pdb")
            ligand_file = os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2")
            
            if not (os.path.exists(protein_file) and os.path.exists(ligand_file)):
                print(f"Files not found for {pdb_id}")
                return None
            
            # Get ID and score
            ligand_id, score, sigma, pl_probs, w  = scoring_single(protein_file, ligand_file, model, device)
            print(f"Processed {pdb_id}: score = {score[0]:.4f}")
            return {
                '#code': ligand_id[0][:4],
                'score': score[0]
            }
        except Exception as e:
            print(f"Error processing {pdb_id}: {str(e)}")
            return None

    # Use joblib for parallel processing
    results = Parallel(n_jobs=-1)(
        delayed(process_single_complex)(pdb_dir) for pdb_dir in pdb_dirs
    )
    # Filter out None results and convert to DataFrame
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)


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

    if args.mode == 'single':
        # Score single complex
        ligand_id, score, *_ = scoring_single(args.protein_file, args.ligand_file, model, device)
        results = pd.DataFrame({
            '#code': [ligand_id[0][:4]],
            'score': [score[0]]
        })
        results.to_csv(args.output, index=False, sep='\t')
        print(f"\nScoring Results:")
        print(f"Ligand ID: {ligand_id}")
        print(f"Predicted Score: {score[0]:.4f}")

    else:  # batch mode
        # Score multiple complexes
        results_df = batch_scoring(args.input_dir, model, device)
        if not results_df.empty:
            results_df.to_csv(args.output, index=False, sep='\t')
            print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

