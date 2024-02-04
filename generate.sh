#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --output=logs/batch_generate_%j.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module purge
module load gcc
module load glib
module load anaconda
source activate symphonynet2

# Batch arguments
scaling=${SCALING}
epoch=${EPOCH}
midifn=${MIDIFN}

echo 'Setting scaling to = '
echo $scaling
echo 'Model epoch is = '
echo $epoch
echo 'MIDI filename = '
echo $midifn

for prime_measures in 5 9 17; do # 1
    EPOCH=$epoch SCALING=$scaling python src/fairseq/gen_batch.py $midifn $prime_measures 0 3
done
