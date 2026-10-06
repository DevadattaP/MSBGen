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

python -m msbgen.cli.evaluate data/processed/kodak \
  --k 3 \
  --models runs/cnn-k3/best.pt runs/unet-k3/best.pt runs/unet-res-k3/best.pt \
  runs/diffusion-k3/best.pt runs/flow-k3/best.pt \
  --lpips --dists --kid --samples 8 --sample-steps 3 \
  --out outputs/eval/kodak-k3-correct --no-thermal-guard

python -m msbgen.cli.plot_eval outputs/eval/kodak-k3-correct \
  --pair flow-k3 unet-res-k3 \
  --out outputs/plots/k3-correct

for s in 1 2 3 5; do
  python -m msbgen.cli.evaluate data/processed/kodak \
    --k 3 \
    --models runs/diffusion-k3/best.pt runs/flow-k3/best.pt \
    --fills --dists --kid --samples 8 --sample-steps "$s" \
    --out "outputs/eval/kodak-k3-gen-s$s" --no-thermal-guard
done

python -m msbgen.cli.plot_steps \
  outputs/eval/kodak-k3-gen-s* \
  outputs/eval/kodak-k3-correct \
  --out outputs/plots/k3-correct

sync
echo "=== Done: $(date)"

echo
echo "=========================================="
echo "Job finished"
echo "End: $(date)"
echo "=========================================="
