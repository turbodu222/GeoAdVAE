#!/bin/bash

#SBATCH --account=biostat
#SBATCH --partition=all-20c96g
#SBATCH --time=48:00:00

#SBATCH --job-name=cm_v6_array
#SBATCH --mem=9G
#SBATCH --cpus-per-task=4

#SBATCH --array=1-63
#SBATCH --error=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup19/slurm/slurm_%A_%a.err
#SBATCH --output=/home/users/%u/kzlinlab/projects/morpho_integration/out/turbo/writeup19/slurm/slurm_%A_%a.out

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cross_modal_cpu_env

# Paths
CSV_FILE="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/patch_seq/parameter_option.csv"
BASE_CONFIG="/home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup19/cross_modal_VAE_v6/configs/attempt_1.yaml"
BASE_OUTPUT="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup19"
CODE_DIR="/home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup19/cross_modal_VAE_v6"

# Get task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Read parameters from CSV (skip header, get line TASK_ID+1)
LINE_NUM=$((TASK_ID + 1))
PARAMS=$(sed -n "${LINE_NUM}p" $CSV_FILE)

# Parse parameters
NO=$(echo $PARAMS | cut -d',' -f1)
GAN_W=$(echo $PARAMS | cut -d',' -f2)
RECON_X_W=$(echo $PARAMS | cut -d',' -f3)
GW_W=$(echo $PARAMS | cut -d',' -f4)
KL_W=$(echo $PARAMS | cut -d',' -f5)
LAMBDA_P=$(echo $PARAMS | cut -d',' -f6)

# Create output directory
OUTPUT_DIR="${BASE_OUTPUT}/cm_VAE_v6_${NO}"
mkdir -p ${OUTPUT_DIR}/slurm
mkdir -p ${OUTPUT_DIR}/configs

# Create temporary config file with modified parameters
TEMP_CONFIG="${OUTPUT_DIR}/configs/config_${NO}.yaml"
cp $BASE_CONFIG $TEMP_CONFIG

# Replace parameters in config file
sed -i "s/^gan_w:.*/gan_w: ${GAN_W}/" $TEMP_CONFIG
sed -i "s/^recon_x_w:.*/recon_x_w: ${RECON_X_W}/" $TEMP_CONFIG
sed -i "s/^gw_w:.*/gw_w: ${GW_W}/" $TEMP_CONFIG
sed -i "s/^kl_w:.*/kl_w: ${KL_W}/" $TEMP_CONFIG
sed -i "s/^lambda_p:.*/lambda_p: ${LAMBDA_P}/" $TEMP_CONFIG

# Print info
echo "============================================"
echo "Running Task ${TASK_ID}: Configuration ${NO}"
echo "Parameters: gan_w=${GAN_W}, recon_x_w=${RECON_X_W}, gw_w=${GW_W}, kl_w=${KL_W}, lambda_p=${LAMBDA_P}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Run training
python ${CODE_DIR}/train.py \
  --config ${TEMP_CONFIG} \
  --output_path ${OUTPUT_DIR} \
  >${OUTPUT_DIR}/slurm/train_${NO}.log 2>&1

echo "Task ${TASK_ID} completed!"
