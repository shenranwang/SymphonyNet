import os
import glob
import re

import pandas as pd
import torch


def accum_evt_instr_matrix(train_dir, epoch):
    files = glob.glob(f"{os.getcwd()}/ckpt/{train_dir}/epoch{epoch}/evt_instr_matrix_*.pt")
    evt_instr_matrices = [torch.load(f, map_location=torch.device('cpu')) for f in files]
    print(len(evt_instr_matrices), evt_instr_matrices[0].shape)
    evt_instr_matrices = torch.stack(evt_instr_matrices)  # Stacks tensors along a new dimension
    print(evt_instr_matrices.shape)
    evt_instr_matrices = evt_instr_matrices.sum(dim=0)
    print(evt_instr_matrices.shape)
    evt_instr_matrix = evt_instr_matrices.sum(dim=0)
    print(evt_instr_matrix.shape)
    torch.save(evt_instr_matrix, f'ckpt/{train_dir}/evt_instr_matrix{epoch}.pt')
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

if __name__ == "__main__":
    for scaling in ["0", "1e6", "1e5", "1e4", "1e7", "1e3", "1e8", "1e2", "1e9", "1e1", "1e10", "1e0", "1e11"]:
        train_dir = f"train_instrument_loss_add_{scaling}_bs128"
        for epoch in [47, 48, 49, 50, 51, 52]:
            print("Scale:", scaling, "Epoch:", epoch)
            accum_evt_instr_matrix(train_dir, epoch)
