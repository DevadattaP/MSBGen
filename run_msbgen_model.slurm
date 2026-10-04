#!/bin/bash
#SBATCH --job-name=msbgen_model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

python -m msbgen.cli.train --preset cnn --name cnn-k3 --k 3 --steps 20000 --device cuda --no-thermal-guard
python -m msbgen.cli.train --preset unet --name unet-k3 --k 3 --steps 20000 --device cuda --no-thermal-guard
python -m msbgen.cli.train --preset unet-res --name unet-res-k3 --k 3 --steps 20000 --device cuda --no-thermal-guard
python -m msbgen.cli.train --preset diffusion --name diffusion-k3 --k 3 --steps 40000 --device cuda --no-thermal-guard
python -m msbgen.cli.train --preset flow --name flow-k3 --k 3 --steps 40000 --device cuda --no-thermal-guard

python -m msbgen.cli.evaluate data/processed/kodak \
      --models runs/cnn-k3/best.pt runs/unet-k3/best.pt runs/unet-res-k3/best.pt \
      runs/diffusion-k3/best.pt runs/flow-k3/best.pt \
      --lpips --dists --kid --samples 8 --sample-steps 3 \
      --out outputs/eval/kodak-k3 --no-thermal-guard

python -m msbgen.cli.plot_eval outputs/eval/kodak-k3 \
      --pair flow-k3 unet-res-k3 --out outputs/plots/k3
sync
echo "=== Done: $(date)"

echo
echo "=========================================="
echo "Job finished"
echo "End: $(date)"
echo "=========================================="
