#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --output=batch_generate.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module purge
module load gcc
module load glib
module load anaconda
source activate symphonynet2

python src/fairseq/gen_batch.py test.mid 5 0 1