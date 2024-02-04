#!/bin/bash
#
while read line;do
	eval "$line"
done < config.sh

while read line;do
	eval "$line"
done < vocab.sh

# for model training
if [ $BPE -eq 0 ]; then
DATA_BIN=linear_${MAX_POS_LEN}_chord_hardloss${IGNORE_META_LOSS}
else
DATA_BIN=linear_${MAX_POS_LEN}_chord_bpe_hardloss${IGNORE_META_LOSS}
fi
DATA_BIN_DIR=data/model_spec/${DATA_BIN}/bin


N_GPU_LOCAL=1
MAX_SENTENCES=2
UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / ${N_GPU_LOCAL}))
echo $UPDATE_FREQ
NN_ARCH=linear_transformer_multi
CHECKPOINT_SUFFIX=${DATA_BIN}_PI${PI_LEVEL}  # for different experiments add suffix here


# Batch arguments
instr_loss_scaling=$1
epoch=$2



FOLD="scratch_instrument_loss_add_${instr_loss_scaling}_bs128"
EPOCHD=$epoch

# EVENT INSTRUMENT MATRIX
# --save-dir ckpt/${FOLD}/tempcp${EPOCHD} --restore-file ckpt/${FOLD}/checkpoint${EPOCHD}_${CHECKPOINT_SUFFIX}.pt
# --cpu --train-subset valid --reset-dataloader

PYTHONWARNINGS="ignore" EPOCH_MODEL=$epoch INSTR_LOSS_SCALING=$instr_loss_scaling fairseq-train 	${DATA_BIN_DIR} \
	--seed ${SEED} \
	--user-dir src/fairseq/linear_transformer \
	--task symphony_modeling --criterion multiple_loss \
	--save-dir ckpt/${FOLD} --restore-file ckpt/baseline.pt \
	--arch ${NN_ARCH} --sample-break-mode complete_doc  --tokens-per-sample ${MAX_POS_LEN} --sample-overlap-rate ${SOR}\
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
	--lr ${PEAK_LR} --lr-scheduler polynomial_decay --warmup-updates ${WARMUP_UPDATES}  --total-num-update ${TOTAL_UPDATES} \
	--dropout 0.1 --weight-decay 0.01 \
	--batch-size ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} \
	--max-update ${TOTAL_UPDATES} --log-format simple --log-interval 10 \
	--checkpoint-suffix _${CHECKPOINT_SUFFIX} \
	--tensorboard-logdir logs/${CHECKPOINT_SUFFIX} \
	--ratio ${RATIO} --evt-voc-size ${SIZE_0} --dur-voc-size ${SIZE_1} --trk-voc-size ${SIZE_2} --ins-voc-size ${SIZE_3} \
	--max-rel-pos ${MAX_REL_POS} --max-mea-pos ${MAX_MEA_POS}  --perm-inv ${PI_LEVEL} \
	2>&1 | tee ${CHECKPOINT_SUFFIX}_part${RECOVER}.log