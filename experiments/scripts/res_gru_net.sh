#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"
NET_NAME=ResidualGRUNet
EXP_DETAIL=default_model
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"
# Make the dir if it not there
mkdir -p $OUT_PATH
# exec &> >(tee -a "$LOG")
# echo Logging output to "$LOG"
# export THEANO_FLAGS="floatX=float32,device=cpu,assert_no_cpu_op='raise'"
export THEANO_FLAGS="floatX=float32,device=cpu"
echo $WEIGHTS
python3 main.py \
      --batch-size 2 \
      --iter 1 \
      --out "$OUT_PATH" \
      --model "$NET_NAME" \
      ${*:1}

python3 main.py \
      --test \
      --batch-size 1 \
      --out "$OUT_PATH" \
      --weights "$OUT_PATH"/weights.npy \
      --model "$NET_NAME" \
      ${*:1}
