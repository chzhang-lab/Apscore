import torch
import torch.utils
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

from data.data import PDBbindDataset, collate_fn, get_stratified_split
from model import APScore, AtomicPairTransformer, ACANet
from utils import EarlyStopping

import argparse
import logging

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)

def huber_loss(pred, target, delta=1.0):
    loss = torch.where(torch.abs(pred - target) < delta,
                      0.5 * (pred - target)**2,
                      delta * (torch.abs(pred - target) - 0.5 * delta))
    return torch.mean(loss)

def compute_loss(score, target, alpha=0.9):
    # Huber Loss
    huber = huber_loss(score, target)
    
    # Pearson correlation loss
    pred_mean = torch.mean(score)
    target_mean = torch.mean(target)
    
    covariance = torch.mean((score - pred_mean) * (target - target_mean))
    pred_std = torch.sqrt(torch.mean((score - pred_mean) ** 2))
    target_std = torch.sqrt(torch.mean((target - target_mean) ** 2))
    
    correlation = covariance / (pred_std * target_std + 1e-5)
    pearson_loss = 1 - correlation
    
    total_loss = alpha * huber + (1-alpha) * pearson_loss    
    return total_loss

def train(model, train_loader, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        model = model.to(device)

        score, _, _, _, _ = model(batch)

        affinity = batch['affinity']
        loss = compute_loss(score, affinity)

        loss = loss / accumulation_steps
        loss.backward()                    # Backward propagation

        # Update parameters every acc_steps
        if (i+1)%accumulation_steps == 0 or (i+1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        del score, loss, affinity
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model = model.to(device)

            score, _, _, _, _ = model(batch)

            affinity = batch['affinity'] 
            loss = compute_loss(score, affinity)

            total_loss += loss.item()

            del score, loss, affinity

    return total_loss / len(val_loader)

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Base data directory
    parser.add_argument("--data_dir", type=str, required=True,
        help="Path to preprocessed dataset directory (e.g., ./dataset/pdbbind_dataset/)")

    # Model training parameters
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    # Network configuration parameters
    parser.add_argument("--atom_feat_dim", type=int, default=23)
    parser.add_argument("--bond_feat_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    
    # Other parameters
    parser.add_argument("--log_file", type=str, default="training.log")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()
    
    # Map data file paths
    file_mappings = {
        'metadata_file': 'metadata.bin',
        'pl_dist_file': 'pl_dist.bin',
        'll_dist_file': 'll_dist.bin',
        'prot_atom_feats_file': 'prot_atom_feats.bin',
        'lig_atom_feats_file': 'lig_atom_feats.bin',
        'prot_neighbor_file': 'prot_neighbor.bin',
        'lig_neighbor_file': 'lig_neighbor.bin',
        'angles_file': 'angles.bin',
        'll_atom1_feats_file': 'll_atom1_feats.bin',
        'll_atom2_feats_file': 'll_atom2_feats.bin',
        'll_neighbor1_file': 'll_neighbor1.bin',
        'll_neighbor2_file': 'll_neighbor2.bin'
    }
    # Set full paths
    for arg_name, file_name in file_mappings.items():
        setattr(args, arg_name, os.path.join(args.data_dir, file_name))
    
    # Generate model name
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_code_name = f"lr{args.lr}_bs{args.batch_size}_{args.atom_feat_dim}_{args.bond_feat_dim}_{args.weight_decay}_{args.dropout_rate}_{args.val_split}_{timestamp}_model.py"
    model_code_path = os.path.join(checkpoint_dir, model_code_name)

    # Read and save model definition file
    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.py')
    with open(model_file, 'r') as f:
        model_code = f.read()
    with open(model_code_path, 'w') as f:
        f.write(model_code)

    model_name = f"lr{args.lr}_bs{args.batch_size}_{args.atom_feat_dim}_{args.bond_feat_dim}_{args.weight_decay}_{args.dropout_rate}_{args.val_split}_{timestamp}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    args.model_path = model_path

    set_seed(args.seed)

    setup_logger(args.log_file)
    logging.info(f"Arguments: {vars(args)}")

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Initialize dataset
    full_dataset = PDBbindDataset(
        args.metadata_file,
        args.pl_dist_file,
        args.ll_dist_file,
        args.prot_atom_feats_file,
        args.lig_atom_feats_file,
        args.prot_neighbor_file,
        args.lig_neighbor_file,
        args.angles_file,
        args.ll_atom1_feats_file,
        args.ll_atom2_feats_file,
        args.ll_neighbor1_file,
        args.ll_neighbor2_file,
    )

    train_dataset, val_dataset = get_stratified_split(
        full_dataset,
        val_split=args.val_split,
        seed=args.seed
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=lambda x: worker_init_fn(x, args.seed)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        worker_init_fn=lambda x: worker_init_fn(x, args.seed)
    )
    
    pl_acanet = ACANet(
        atom_feat_dim = args.atom_feat_dim,
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
        atom_feat_dim = args.atom_feat_dim,
        bond_feat_dim=args.bond_feat_dim,
        hidden_dim=args.hidden_dim,
    )
    
    model = APScore(
        pl_acanet,
        aptransformer,
        ll_acanet,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=args.patience, mode='lower', filename=args.model_path)

    train_losses = []
    val_losses = []

    effective_batch_size = args.batch_size * args.accumulation_steps
    logging.info(f"Effective batch size: {effective_batch_size} (batch_size={args.batch_size} × accumulation_steps={args.accumulation_steps})")

    # Training loop
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, args.device, accumulation_steps=args.accumulation_steps)
        val_loss = evaluate(model, val_loader, args.device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logging.info(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.getLogger().handlers[0].flush()

        if early_stopping.step(val_loss, model):
            logging.info("Early stopping triggered")
            break

        if (epoch + 1) % 2 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss (Epoch {epoch+1})\n{model_name}')
            plt.legend()
            plt.savefig('loss_plot.png')
            plt.close()

    # Load best model
    early_stopping.load_checkpoint(model)

    logging.info(f"Best validation score: {early_stopping.best_score:.4f}")

    # Final loss curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Epoch {epoch+1})')
    plt.legend()
    plt.savefig('final_loss_plot.png')
    plt.close()
    logging.info("Final loss plot saved as final_loss_plot.png")

if __name__ == "__main__":
    main()
