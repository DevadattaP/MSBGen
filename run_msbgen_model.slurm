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
echo "Starting MSBGen RGB run..."
date

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Usage: sbatch --export=ALL,KS="1" run_msbgen_model.slurm   (one k per job fits the time limit)
KS=${KS:-"1 2 3"}
R=runs/rgb
TRAIN=data/processed/div2k-train-rgb
VAL=data/processed/div2k-valid-rgb
TEST=data/processed/kodak-rgb
COMMON="--train $TRAIN --val $VAL --out-dir $R --device cuda --no-thermal-guard"

# RGB data: (H, W, 3) arrays in data/processed/<name>-rgb (gray data is untouched)
python -m msbgen.cli.prepare_data div2k kodak --color rgb

# Train one model; continue from last.pt when the job was cut off
train_model() {  # name, then extra train args
  local name=$1; shift
  local resume=""
  [ -f "$R/$name/last.pt" ] && resume="--resume $R/$name/last.pt"
  python -m msbgen.cli.train --name "$name" $COMMON $resume "$@"
}

# Sequential bit-plane CNNs (paper approach), one per missing plane, 3 channels each (k independent)
for b in 1 2 3 4 5 6 7; do
  if [ ! -f "$R/bitdepth-d4/bit_0${b}.pt" ]; then
    python -m msbgen.cli.bitdepth train --input-bits "$b" --train $TRAIN --val $VAL \
      --out $R/bitdepth-d4 --augment --device cuda --no-thermal-guard
  fi
done

for k in $KS; do
  if [ "$k" -eq 1 ]; then s=""; else s="-k${k}"; fi

  train_model cnn${s}              --preset cnn --k "$k"
  train_model unet${s}             --preset unet --k "$k"
  train_model unet-res${s}         --preset unet-res --k "$k"
  train_model normalizing-flow${s} --preset normalizing-flow --k "$k"
  train_model flow${s}             --preset flow --k "$k"
  train_model diffusion${s}        --preset diffusion --k "$k"
  train_model vit${s}              --preset vit --k "$k"
  train_model hybrid${s}           --preset hybrid --k "$k"
  train_model vit-hybrid${s}       --preset vit-hybrid --k "$k"

  python -m msbgen.cli.evaluate $TEST --k "$k" \
    --models $R/cnn${s}/best.pt $R/unet${s}/best.pt $R/unet-res${s}/best.pt \
             $R/vit${s}/best.pt $R/normalizing-flow${s}/best.pt $R/flow${s}/best.pt \
             $R/diffusion${s}/best.pt $R/hybrid${s}/best.pt $R/vit-hybrid${s}/best.pt \
    --bitdepth-models $R/bitdepth-d4 --hybrid-ablation \
    --lpips --dists --kid --samples 8 --sample-steps 3 --save-images 4 \
    --out outputs/rgb/eval/kodak-k${k} --device cuda --no-thermal-guard

  python -m msbgen.cli.plot_eval outputs/rgb/eval/kodak-k${k} \
    --pair vit-hybrid${s} hybrid${s} --out outputs/rgb/plots/k${k}

  python -m msbgen.cli.plot_runs $R/cnn${s} $R/unet${s} $R/unet-res${s} $R/vit${s} \
    $R/normalizing-flow${s} $R/flow${s} $R/diffusion${s} $R/hybrid${s} $R/vit-hybrid${s} \
    --out outputs/rgb/plots/runs-k${k}
done

# Bit-depth recovery on its own (start bits 1..7 -> 8), RGB
python -m msbgen.cli.bitdepth evaluate --models $R/bitdepth-d4 --test $TEST \
  --start-bits 1 2 3 4 5 6 7 --save-images --out outputs/rgb/bitdepth/kodak --device cuda
python -m msbgen.cli.bitdepth plot outputs/rgb/bitdepth/kodak/evaluation.csv \
  --out outputs/rgb/bitdepth/kodak/recovery.png

sync
echo "=== Done: $(date)"

echo
echo "=========================================="
echo "Job finished"
echo "End: $(date)"
echo "=========================================="
