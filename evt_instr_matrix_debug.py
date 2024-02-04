import torch
import json

from src.fairseq.gen_utils import music_dict
from src.encoding import bpe_str2int, ison, ispitch

MAX_POS_LEN = 4096
IGNORE_META_LOSS = 1

BPE = "_bpe"
# BPE = ""
DATA_BIN=f"linear_{MAX_POS_LEN}_chord{BPE}_hardloss{IGNORE_META_LOSS}"
DATA_VOC_DIR=f"data/model_spec/{DATA_BIN}/vocabs/"
music_dict.load_vocabs_bpe(DATA_VOC_DIR, 'data/bpe_res/' if BPE == '_bpe' else None)

instrument_ranges = json.load(open("midi_pitch_ranges.json", "r"))
instrument_ranges = list(instrument_ranges.values())

def mask_instr_range():
    # TODO: Implement proper logging and plotting for losses, and possibly try plotting the event-instrument matrix
    mask = torch.ones((2,1125,133), device='cpu')  #  if torch.cuda.is_available() else 'cpu'
    mask[:,:4,:] = 0  # special tokens
    mask[:,:,:4] = 0  # special tokens
    for i in range(len(instrument_ranges)):
        low, high = instrument_ranges[i]
        
        low, high = music_dict.str2int[low], music_dict.str2int[high]
        # print(instrument_ranges[i], low, high)
        mask[:,low:high,i+4] = 0  # note tokens
        for k in music_dict.vocabs[0]:
            tok = music_dict.vocabs[0][k]
            if ison(tok):
                notes = bpe_str2int(tok)
                if notes[0] == 1:  # bpe token
                    f, l = notes[1], notes[-1]
                    if low <= f and high >= l:
                        mask[:,int(k),i+4] = 0
            if not ispitch(tok) and not ison(tok):  # non pitch or bpe tokens, must not be counted in loss
                mask[:,int(k),i+4] = 0
    mask[:,:,-1] = 0  # extra instr token
    return mask
    

def check_mask_instr_range(mask):
    for i in range(2):
        sample = mask[i]
        assert torch.all(sample[:4, :] == 0)
        assert torch.all(sample[:, :4] == 0)
        for k in music_dict.vocabs[0]:
            tok = music_dict.vocabs[0][k]
            if not ison(tok) and not ispitch(tok):
                # print(k, int(k), tok, "non pitch or bpe token")
                # print(sample[int(k), :])
                assert torch.all(sample[int(k), :] == 0)
        # for evt in range(1125):
        #     for instr in range(133):
        #         if ispitch(tok)

create = False

if create:
    mask_instr = mask_instr_range()
    check_mask_instr_range(mask_instr)
    torch.save(mask_instr, "evt_instr_matrix.pt")
else:
    mask_instr = torch.load("evt_instr_matrix.pt")
    check_mask_instr_range(mask_instr)

for k in music_dict.vocabs[0]:
    tok = music_dict.vocabs[0][k]
    
# for event in range(1125):
#     print(music_dict.vocabs[0])
    # for instr in range(133):
    
# print(mask_instr[0].shape, 1125*133, torch.sum(mask_instr[0]))
