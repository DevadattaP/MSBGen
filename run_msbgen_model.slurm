#!/bin/bash
#SBATCH --job-name=msbgen_model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

echo "=========================================="
echo "MSBGen model job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Load the HPC Python module
module load python-3.12.8

# Activate your HPC environment
source ~/venv_hpc/bin/activate

# Make sure the project directory is the working directory
cd ~/MSBGen


echo "Python:"
which python
python --version

echo
echo "GPU:"
nvidia-smi

echo
echo "PyTorch:"
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

echo
echo "Starting MSBGen model run..."
date

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

WORKERS=$((SLURM_CPUS_PER_TASK - 2))
echo "Using $WORKERS DataLoader workers"

for bits in 1 2 3 4 5 6 7; do
  python -m msbgen.cli.bitdepth train \
    --input-bits "$bits" \
    --train data/processed/div2k-train \
    --val data/processed/div2k-valid \
    --out runs/bitdepth-d4 \
    --bit-depth 8 \
    --depth 4 \
    --epochs 30 \
    --batch-size 128 \
    --patch-size 48 \
    --lr 1e-3 \
    --lr-drop-epoch 16 \
    --workers "$WORKERS" \
    --device cuda \
    --augment --no-thermal-guard
done

python -m msbgen.cli.bitdepth evaluate \
  --models runs/bitdepth-d4 \
  --test data/processed/kodak \
  --start-bits 1 2 3 \
  --end-bits 8 \
  --bit-depth 8 \
  --device cuda \
  --save-images \
  --out outputs/bitdepth/kodak

python -m msbgen.cli.bitdepth plot \
  outputs/bitdepth/kodak/evaluation.csv \
  --out outputs/bitdepth/kodak/recovery.png

for k in 1 2 3; do
  if [ "$k" -eq 1 ]; then
    suffix=""
  else
    suffix="-k${k}"
  fi

  python -m msbgen.cli.evaluate \
    data/processed/kodak \
    --k "$k" \
    --models \
      "runs/cnn${suffix}/best.pt" \
      "runs/unet${suffix}/best.pt" \
      "runs/unet-res${suffix}/best.pt" \
      "runs/diffusion${suffix}/best.pt" \
      "runs/flow${suffix}/best.pt" \
    --out "outputs/eval/kodak-all-k${k}" \
    --samples 4 \
    --save-images 4 \
    --device cuda --no-thermal-guard
done

sync
echo "=== Done: $(date)"

echo
echo "=========================================="
echo "Job finished"
echo "End: $(date)"
echo "=========================================="
