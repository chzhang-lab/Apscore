# APScore: An Interatomic Distance Probability Model for Physically Realistic Protein–Ligand Interaction Scoring

APScore is an atomic-resolution protein–ligand scoring function built around an **Interatomic Distance Probability (IADP)** model. Rather than representing a complex primarily as a global molecular graph, APScore models localized protein–ligand atom pairs using their interatomic distances and angular geometries. The IADP model parameterizes interaction probabilities in a physically informed form inspired by the distance dependence of the Lennard–Jones potential and the Boltzmann distribution.

APScore combines the **IADP model** with a **line-of-sight steric occlusion filter** and an **Atomic Context-Aware Network (ACA-Net)**. This design focuses the model on physically accessible local contacts, reduces reliance on molecular scaffold memorization, and provides interpretable atom-level interaction parameters for structure-based drug discovery.

## Key Features
* **Physics-guided IADP model:** Learns continuous interatomic distance probability distributions rather than directly learning an unconstrained black-box scoring function, providing an explicit representation of distance-dependent interactions.
* **Local atomic representation with ACA-Net:** Decomposes protein–ligand complexes into contact atom pairs and their local chemical contexts, reducing dependence on global molecular topology and supporting out-of-distribution generalization.
* **Line-of-sight steric filtering:** Identifies and excludes spatially adjacent atom pairs that are geometrically occluded by intervening atoms, restricting the interaction representation to physically accessible contacts.
* **Explicit atomic interpretability:** Predicts optimal interaction distances (σ) and interaction weights (w) for individual atom pairs, enabling analysis of distinct interaction patterns such as hydrogen bonding and hydrophobic contacts.

---

## Project Structure

```text
Apscore/
├── feats/                         # Feature extraction & geometric filtering
│   ├── atom_feats.py              # Atom-level feature encoder (RDKit)
│   ├── atom_pairs.py              # Atom-pair extraction and graph construction
│   ├── mol2inter_feats.py         # Main entry: Complex → local context subgraphs
│   ├── shielded_detection.py      # Line-of-sight steric shielding detection algorithm
│   ├── target_opt.py              # Protein hydrogen addition wrapper (pdb2pqr)
│   ├── dataset2inter_feats.py     # PDBbind batch preprocessing
│   └── pdbbind2inter_feats.sh     # SLURM job script for preprocessing
│
├── data/                          # Data loading pipeline
│   └── data.py                    # PyTorch Dataset classes + collate functions
│
├── model/                         # Neural network architecture
│   ├── model.py                   # ACA-Net + Transformer Module + IADP scoring
│   ├── train.py                   # Training loop & composite loss (Huber + Pearson)
│   ├── utils.py                   # Metric tracking + Early stopping
│   └── train.sh                   # SLURM job script for training
│
├── inference/                     # Application modules
│   ├── scoring.py                 # Absolute binding affinity prediction
│   ├── docking.py                 # Docking pose scoring and ranking
│   ├── screening.py               # Large-scale virtual screening
│   ├── batch_scoring.sh
│   ├── batch_docking.sh
│   └── batch_screening.sh
│
├── environment.yml                # Conda environment configuration
└── README.md
```

## Installation

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage the software environment and dependencies, including RDKit and pdb2pqr.

```bash
# Clone repository
git clone https://github.com/chzhang-lab/Apscore.git
cd Apscore

# Create the conda environment and install all dependencies automatically
conda env create -f environment.yml

# Activate the environment
conda activate apscore
```

---

## Dataset Preparation

You can either download the preprocessed datasets directly or process the raw data from scratch.

### Option A: Download Precomputed Data (⚡ Quick Start)
To facilitate rapid reproduction, fully preprocessed training files for both the **standard PDBbind** and **PDBbind CleanSplit** datasets are publicly available on Zenodo: [https://doi.org/10.5281/zenodo.21317993](https://doi.org/10.5281/zenodo.21317993).

Download the archives, extract them into the local `dataset/` directory, and proceed directly to the **Model Training** section.

### Option B: Process from Scratch
To process the raw PDBbind datasets manually, follow the pipeline below.

**1. Prepare Raw Data**
Organize your complex datasets (e.g., PDBbind V2020) with the following structure:
```text
PDBind/
├── general_set/
│   └── {pdb_id}/
│       ├── {pdb_id}_pocket.pdb         # Original protein pocket
│       ├── {pdb_id}_ligand.mol2        # Ligand structure (3D conformer with hydrogens)
└── INDEX_general_PL_data.2020          # Binding affinity index
```

**2. Add Hydrogens to Proteins**
Accurate hydrogen placement is important for the angular and distance features used by APScore:
```bash
python feats/target_opt.py --input_dir /path/to/PDBind/general_set/
```
*(This invokes `pdb2pqr30` with the AMBER force field to generate `{pdb_id}_pocket_addh.pdb`.)*

**3. Run Preprocessing & Steric Filtering**
Extract local context features and apply geometric steric occlusion filtering:
```bash
python feats/dataset2inter_feats.py     --input_dirs /path/to/PDBind/general_set/     --output_prefix ./dataset/pdbbind_dataset/     --affinity_file /path/to/INDEX_general_PL_data.2020     --coreset_path /path/to/PDBind/coreset/     --batch_size 500
```

---

## Model Training

APScore is trained using a joint loss comprising Huber loss and a Pearson correlation penalty. Ensure that the preprocessed data, whether downloaded or generated locally, is placed in the `dataset/` directory.

```bash
# To train on the standard PDBbind dataset
python model/train.py --data_dir ./dataset/pdbbind_dataset/

# To train on the de-duplicated PDBbind CleanSplit dataset
python model/train.py --data_dir ./dataset/cleansplit_dataset/
```

**Outputs:**
- `checkpoints/best_model.pth`: Model weights from the best validation epoch, selected by early stopping.
- `training.log` and `final_loss_plot.png`: Training and optimization diagnostics.

### SLURM Batch Job
```bash
sbatch model/train.sh
```

---

## Inference & Applications

During inference, APScore applies a distance-based collision penalty when an interacting atom pair falls below the specified 0.55σ cutoff, helping suppress physically implausible conformations in structure-based applications.

### 1. Binding Affinity Scoring
Predict the binding affinity of protein–ligand complexes.

```bash
python inference/scoring.py     --mode single     -p /path/to/protein_pocket_addh.pdb     -l /path/to/ligand.mol2     -m checkpoints/best_model.pth     -o result.dat
```

### 2. Docking Pose Recognition
Rank computationally generated ligand poses and identify near-native binding conformations.

```bash
python inference/docking.py     -pb /path/to/protein_base/     -lb /path/to/ligand_base/     -m checkpoints/best_model.pth     -o ./docking_results/
```

### 3. Virtual Screening
Rank compounds against a target protein, such as the STING transmembrane ion channel.

```bash
python inference/screening.py     -pb /path/to/protein_base/     -lb /path/to/ligand_base/     -m checkpoints/best_model.pth     -o ./screening_results/
```

---

## Input Formats
- **Proteins:** `.pdb` format containing three-dimensional coordinates and added hydrogen atoms.
- **Ligands:** `.mol2` or `.sdf` format containing valid three-dimensional conformers with explicit hydrogens.

---

## Citation & Paper

The manuscript describing APScore and its application to the discovery of novel STING transmembrane ion channel modulators is currently under review.

> *An interatomic distance probability model enables physically realistic and interpretable protein–ligand interaction prediction.* (Under review, 2026)

Citation information will be updated upon publication.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
APScore was developed at the College of Biology, Hunan University. We acknowledge the use of the [PDBbind](https://www.pdbbind-plus.org.cn/) dataset and [pdb2pqr](https://github.com/Electrostatics/pdb2pqr) for biomolecular structure preparation.
