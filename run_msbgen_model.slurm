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

R=$HOME/MSBGen/runs
E=$HOME/MSBGen/outputs/eval

echo "=== Step 1: train all k=2 models: $(date)"
for p in cnn unet unet-res flow diffusion; do
    python -m msbgen.cli.train --preset $p --k 2 --name $p-k2 --no-thermal-guard
    ls -la $R/$p-k2
done

echo "=== Step 2: main evaluation (all models + fills, 3 steps for generative): $(date)"
python -m msbgen.cli.evaluate data/processed/kodak --k 2 --models $R/*-k2/best.pt \
    --lpips --dists --kid --samples 8 --out $E/kodak-k2 --no-thermal-guard
ls -la $E/kodak-k2

echo "=== Step 3: step sweep: $(date)"
for s in 1 2 3 5 10 20 50; do
    python -m msbgen.cli.evaluate data/processed/kodak --k 2 \
        --models $R/flow-k2/best.pt $R/diffusion-k2/best.pt --fills \
        --dists --kid --samples 8 --sample-steps $s --out $E/kodak-k2-gen-s$s --no-thermal-guard
    ls -la $E/kodak-k2-gen-s$s
done

echo "=== Step 4: plots: $(date)"
P=$HOME/MSBGen/outputs/plots/k2
python -m msbgen.cli.plot_runs $R/cnn-k2 $R/unet-k2 $R/unet-res-k2 $R/flow-k2 $R/diffusion-k2 --out $P
python -m msbgen.cli.plot_eval $E/kodak-k2 --pair flow-k2 unet-res-k2 --out $P
python -m msbgen.cli.plot_steps $E/kodak-k2-gen-s* $E/kodak-k2 --out $P
ls -la $P
sync
echo "=== Done: $(date)"

echo
echo "=========================================="
echo "Job finished"
echo "End: $(date)"
echo "=========================================="
