# Light-HTINet-DNS

Pure topological graph learning with dynamic negative sampling for
herb-target interaction (HTI) prediction.

This is the official implementation of:

> Yan X, et al. Eliminating Semantic Noise in Herb-Target Prediction:
> A Pure Topological Graph Approach with Hard Negative Mining.
> *Journal of Cheminformatics*, 2026.

## Environment

- Python 3.8+
- PyTorch 1.12+
- Install dependencies: `pip install -r requirements.txt`

## Data

Place the following files in the `data/` directory:

- `training_set.npy`
- `val_set.npy`
- `testing_set.npy`
- `set_all.npy`
- `name2id.txt`

These follow the HTINet2 benchmark (Duan et al., *Briefings in Bioinformatics* 2024).
Statistics: 563 herbs, 2106 genes, 30,626 training edges, 4,022 validation edges,
3,354 test edges.

## Quick Start

### Train the main model (DNS K=12, margin=0.5)

```bash
python run_seeds.py
```

This trains three seeds (1, 42, 2024) and saves checkpoints to
`outputs/seed_{seed}/s_seed{seed}/`.

### Evaluate

```bash
python batch_eval.py
```

This runs validation-based epoch selection on each seed and reports
mean ± std. Expected result: **HR@10 ≈ 0.562 ± 0.003,
NDCG@10 ≈ 0.473 ± 0.002**.

### Single seed training / evaluation

```bash
# Train
python train.py --seed 1 --output_dir ./outputs/seed_1

# Evaluate
python test.py --seed 1 --model_dir ./outputs/seed_1/s_seed1
```

## Reproducing Ablation Studies

### Varying DNS candidate pool size K

```bash
python train.py --seed 1 --dns_k 1  --output_dir ./outputs/abl_k1
python train.py --seed 1 --dns_k 3  --output_dir ./outputs/abl_k3
python train.py --seed 1 --dns_k 5  --output_dir ./outputs/abl_k5
python train.py --seed 1 --dns_k 8  --output_dir ./outputs/abl_k8
python train.py --seed 1 --dns_k 12 --output_dir ./outputs/abl_k12
```

### Varying margin

```bash
python train.py --seed 1 --dns_k 12 --margin 0.0 --output_dir ./outputs/abl_m0
python train.py --seed 1 --dns_k 12 --margin 0.5 --output_dir ./outputs/abl_m05
python train.py --seed 1 --dns_k 12 --margin 0.8 --output_dir ./outputs/abl_m08
python train.py --seed 1 --dns_k 12 --margin 1.0 --output_dir ./outputs/abl_m1
```

### Case study visualization

```bash
python predict.py --seed 1 --epoch 380 --model_dir ./outputs/seed_1/s_seed1
```

## Project Structure

```
Light-HTINet-DNS/
├── train.py          # Training loop (DNS + BPR margin loss)
├── test.py           # Validation-based epoch selection + one-shot test eval
├── batch_eval.py     # Run test.py over all seeds and aggregate mean ± std
├── predict.py        # Case study: top-K target prediction & visualization
├── data_utils.py     # EnhancedHTI model, BPR dataset, evaluation metrics
├── run_seeds.py      # Train seeds 1, 42, 2024 sequentially
├── requirements.txt
└── data/
    ├── training_set.npy
    ├── val_set.npy
    ├── testing_set.npy
    ├── set_all.npy
    └── name2id.txt
```

## Citation

If you use this code, please cite:

```
Yan X, et al. Eliminating Semantic Noise in Herb-Target Prediction:
A Pure Topological Graph Approach with Hard Negative Mining.
Journal of Cheminformatics, 2026.
```
