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

for k in 1 2 3; do
  if [ "$k" -eq 1 ]; then s=""; else s="-k${k}"; fi
  python -m msbgen.cli.evaluate data/processed/kodak --k "$k" \
    --models runs/cnn${s}/best.pt runs/unet${s}/best.pt runs/unet-res${s}/best.pt \
             runs/diffusion${s}/best.pt runs/flow${s}/best.pt runs/hybrid${s}/best.pt \
    --bitdepth-models runs/bitdepth-d4 --hybrid-ablation \
    --lpips --dists --kid --samples 8 --sample-steps 3 --save-images 4 \
    --out outputs/eval/kodak-final-k${k} --device cuda --no-thermal-guard

  python -m msbgen.cli.plot_eval outputs/eval/kodak-final-k${k} \
    --pair hybrid${s} bitdepth-d4 --out outputs/plots/final-k${k}
done

sync
echo "=== Done: $(date)"

echo
echo "=========================================="
echo "Job finished"
echo "End: $(date)"
echo "=========================================="
