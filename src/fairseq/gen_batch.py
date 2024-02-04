import os, sys, time

MAX_POS_LEN = 4096
PI_LEVEL = 2
IGNORE_META_LOSS = 1

BPE = "_bpe"
# BPE = ""
DATA_BIN=f"linear_{MAX_POS_LEN}_chord{BPE}_hardloss{IGNORE_META_LOSS}"
CHECKPOINT_SUFFIX=f"{DATA_BIN}_PI{PI_LEVEL}"
DATA_BIN_DIR=f"data/model_spec/{DATA_BIN}/bin/"
DATA_VOC_DIR=f"data/model_spec/{DATA_BIN}/vocabs/"
from gen_utils import process_prime_midi, gen_one, get_trk_ins_map, get_note_seq, note_seq_to_midi_file, music_dict
music_dict.load_vocabs_bpe(DATA_VOC_DIR, 'data/bpe_res/' if BPE == '_bpe' else None)


from fairseq.models import FairseqLanguageModel

scale = os.environ['SCALING']
cp = os.environ['EPOCH']

MODEL_FOLDER = f'scratch_instrument_loss_add_{scale}_bs128'
custom_lm = FairseqLanguageModel.from_pretrained('.',
    checkpoint_file=f'ckpt/{MODEL_FOLDER}/checkpoint{cp}_linear_4096_chord_bpe_hardloss1_PI2.pt',
    data_name_or_path=DATA_BIN_DIR, 
    user_dir="src/fairseq/linear_transformer_inference")
print(f'Generation using model: ckpt/{MODEL_FOLDER}/checkpoint{cp}_linear_4096_chord_bpe_hardloss1_PI2.pt')

m = custom_lm.models[0]
m.cuda()
m.eval()


GEN_DIR = f'generated/{MODEL_FOLDER}/'
os.makedirs(GEN_DIR, exist_ok=True)


def seed_everything(seed=12345, mode='balanced'):
    # ULTIMATE random seeding for either full reproducibility or a balanced reproducibility/performance.
    # In general, some operations in cuda are non deterministic to make them faster, but this can leave to
    # small differences in several runs.
    #
    # So as of 21.10.2021, I think that the best way is to use the balanced approach during exploration
    # and research, and then use the full reproducibility to get the final results (and possibly share code)
    #
    # References:
    # https://pytorch.org/docs/stable/notes/randomness.html
    #
    # Args:
    #   -- seed = Random seed
    #   -- mode {'balanced', 'deterministic'}

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode == 'balanced':
        torch.backends.cudnn.deterministic = False   # if set as true, dilated convs are really slow
        torch.backends.cudnn.benchmark = True  # True -> better performance, # False -> reproducibility
    elif mode == 'deterministic':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Throws error:
        # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)`
        # or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS
        # and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable
        # before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
        # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

        # torch.use_deterministic_algorithms(True)




if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('usage: python src/fairseq/gen_batch.py <prime_midi_file> <prime_measure_count> <prime_chord_count> <gen_count>')
        exit(0)
    midi_name = sys.argv[1].split('/')[-1][:-4]
    max_measure_cnt = int(sys.argv[2])
    max_chord_measure_cnt = int(sys.argv[3])
    prime, ins_label = process_prime_midi(f"midis/{sys.argv[1]}", max_measure_cnt, max_chord_measure_cnt)
    gen_cnt = int(sys.argv[4])
    for i in range(gen_cnt):
        while(True):
            try:
                generated, ins_logits = gen_one(m, prime, MIN_LEN = 1024)
                break
            except Exception as e:
                print(e)
                continue
        trk_ins_map = get_trk_ins_map(generated, ins_logits)
        note_seq = get_note_seq(generated, trk_ins_map)
        #print(f'{len(note_seq)} notes generated.')
        #print(note_seq)
        timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime()) 
        note_seq_to_midi_file(note_seq, f'{GEN_DIR}{midi_name}_prime{max_measure_cnt}_chord{max_chord_measure_cnt}_gen{i}_{scale}_{cp}.mid')
    print("FINISHED GENERATING!")