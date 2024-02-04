#!/bin/bash

# TRAINING
for scale in 5e6 8e6 2e7 5e7 8e7 2e8 5e8 # 0 1e6 1e5 1e4 1e7 1e3 1e8 1e2 1e9 1e1 1e10 1e0 1e11
do
    INSTR_LOSS_SCALING=$scale sbatch train.sh
done

# EVENT INSTRUMENT MATRIX
# for scale in 0 5e6 8e6 2e7 5e7 8e7 2e8 5e8 1e6 1e5 1e4 1e7 1e3 1e8 1e2 1e9 1e1 1e10 1e0 1e11; do  # 0
#     for epoch in 47 48 49 50 51 52; do
#         INSTR_LOSS_SCALING=$scale EPOCH=$epoch sbatch train.sh
#     done
# done