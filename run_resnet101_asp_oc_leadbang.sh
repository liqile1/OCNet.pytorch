# check the enviroment info
nvidia-smi
# pytorch 04
PYTHON="/root/miniconda3/bin/python"


#network config
NETWORK="resnet101"
METHOD="asp_oc_dsn"
DATASET="leadbang_train"

#training settings
LEARNING_RATE=1e-2
WEIGHT_DECAY=5e-4
START_ITERS=0
MAX_ITERS=5000
BATCHSIZE=4
INPUT_SIZE='512,512'
USE_CLASS_BALANCE=True
USE_OHEM=False
OHEMTHRES=0.7
OHEMKEEP=0
USE_VAL_SET=False
USE_EXTRA_SET=False

# replace the DATA_DIR with your folder path to the dataset.
DATA_DIR='./dataset/leadbang'
DATA_LIST_PATH='t'
RESTORE_FROM='./pretrained_model/resnet101-imagenet.pth'

# Set the Output path of checkpoints, training log.
TRAIN_LOG_FILE="./log/log_train/log_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}"	
SNAPSHOT_DIR="./checkpoint/snapshots_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"


########################################################################################################################
#  Training
########################################################################################################################
python train_leadbang.py --network $NETWORK --method $METHOD --random-mirror --random-scale --gpu 0,1,2,3 --batch-size $BATCHSIZE \
  --snapshot-dir $SNAPSHOT_DIR  --num-steps $MAX_ITERS --ohem $USE_OHEM --data-list $DATA_LIST_PATH --weight-decay $WEIGHT_DECAY \
  --input-size $INPUT_SIZE --ohem-thres $OHEMTHRES --ohem-keep $OHEMKEEP --use-val $USE_VAL_SET --use-weight $USE_CLASS_BALANCE \
  --snapshot-dir $SNAPSHOT_DIR --restore-from $RESTORE_FROM --start-iters $START_ITERS --learning-rate $LEARNING_RATE  \
  --use-extra $USE_EXTRA_SET --dataset $DATASET --data-dir $DATA_DIR 

