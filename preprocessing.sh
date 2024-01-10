#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=batch_run.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

python3 src/preprocess/preprocess_midi.py
python3 src/preprocess/get_bpe_data.py
python3 src/fairseq/make_data.py