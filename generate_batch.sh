#!/bin/bash

# for midiname in debussy.mid dvorak.mid test.mid stringquartet.mid pinkpanther.mid abba.mid rainbow.mid dreamy.mid; do
#     SCALING=$scale EPOCH=$epoch MIDIFN=$midiname sbatch generate.sh
# done

for scale in 5e7 0; do  # 1e6 1e5 1e4 1e7 1e3 1e8 1e2 1e9 1e1 1e10 1e0 1e11
    for epoch in 3; do  # 48 52
        for midiname in debussy.mid dvorak.mid test.mid stringquartet.mid pinkpanther.mid abba.mid rainbow.mid dreamy.mid; do
            SCALING=$scale EPOCH=$epoch MIDIFN=$midiname sbatch generate.sh
        done
    done
done