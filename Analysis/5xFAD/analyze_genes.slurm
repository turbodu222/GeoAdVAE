#!/bin/bash

#SBATCH --account=biostat
#SBATCH --partition=all-20c96g
#SBATCH --time=3:00:00

#SBATCH --job-name=gex_ig
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

#SBATCH --error=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup16/cm_VAE_v6_implement_v3/slurm/gex_ig_%j.err
#SBATCH --output=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup16/cm_VAE_v6_implement_v3/slurm/gex_ig_%j.out

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cross_modal_cpu_env

# Run GEX gene importance analysis
python /home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup16/cross_modal_VAE_v6_implement/analyze_gex_genes_ig.py \
  --config /home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup16/cross_modal_VAE_v6_implement/configs/attempt_1.yaml \
  --checkpoint_dir /home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup16/cm_VAE_v6_implement_v3/outputs/attempt_1/checkpoints \
  --output_dir /home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup16/cm_VAE_v6_implement_v3/outputs/attempt_1/gex_ig_analysis \
  --n_steps 50 \
  --batch_size 32 \
  --baseline_type zeros \