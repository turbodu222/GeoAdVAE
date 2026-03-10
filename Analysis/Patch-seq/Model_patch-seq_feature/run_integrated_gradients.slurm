#!/bin/bash

#SBATCH --account=biostat
#SBATCH --partition=all-20c96g
#SBATCH --time=6:00:00

#SBATCH --job-name=ig_analysis
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

#SBATCH --error=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/slurm/ig_analysis_%j.err
#SBATCH --output=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/slurm/ig_analysis_%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cross_modal_cpu_env

# Print environment info
echo "========================================"
echo "Starting Integrated Gradients Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"
echo ""

conda env list
python --version
echo ""

# Check if captum is installed
echo "Checking for Captum library..."
python -c "import captum; print(f'Captum version: {captum.__version__}')" || echo "Warning: Captum not found. Installing..."

# Install captum if not present
python -c "import captum" 2>/dev/null || pip install captum

echo ""
echo "Starting analysis..."
echo ""

# Run Integrated Gradients analysis
python /home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup23/cross_modal_VAE_v6/integrated_gradients_gene_analysis.py \
  --config /home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup23/cross_modal_VAE_v6/configs/attempt_1.yaml \
  --checkpoint_dir /home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/outputs/attempt_1/checkpoints \
  --output_dir /home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/outputs/attempt_1 \
  --gene_expression_path /home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_norm_paired_2000.csv \
  --baseline_type zeros \
  --n_steps 100

echo ""
echo "========================================"
echo "Analysis completed!"
echo "========================================"