#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_instrument_loss_add_bs128_%j.out
#SBATCH --constraint=volta|ampere

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module purge
module load gcc
module load glib
module load cuda
module load anaconda
source activate symphonynet2

nvcc --version

# Batch arguments
instr_loss_scaling=${INSTR_LOSS_SCALING}

echo 'Setting instr_loss_scaling to = '
echo $instr_loss_scaling

sh train_linear_chord.sh $instr_loss_scaling