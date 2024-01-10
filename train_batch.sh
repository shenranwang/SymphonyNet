#!/bin/bash

for scale in 0 1e5 1e6 1e4 1e7 1e3 1e8 1e2 1e9 1e1 1e10 1e0 1e11
do
    INSTR_LOSS_SCALING=$scale sbatch train.sh
done