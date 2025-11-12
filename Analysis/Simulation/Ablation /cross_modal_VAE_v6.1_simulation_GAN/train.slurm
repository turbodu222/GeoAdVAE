#!/bin/bash

#SBATCH --account=biostat
#SBATCH --partition=all-20c96g
#SBATCH --time=48:00:00

#SBATCH --job-name=cm_v6
#SBATCH --mem=9G
#SBATCH --cpus-per-task=4

#SBATCH --error=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup18/cm_VAE_v6_simu_GAN/slurm/slurm_%j.err
#SBATCH --output=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup18/cm_VAE_v6_simu_GAN/slurm/slurm_%j.out

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cross_modal_cpu_env

# Optional: print environment info
conda env list
python --version
conda list

# Run training
python /home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup18/cross_modal_VAE_v6.1_simulation_GAN/train.py \
  --config /home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup18/cross_modal_VAE_v6.1_simulation_GAN/configs/attempt_1.yaml \
  --output_path /home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup18/cm_VAE_v6_simu_GAN \
  >/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup18/cm_VAE_v6_simu_GAN/slurm/cross_modal_train-$SLURM_JOB_ID.txt
