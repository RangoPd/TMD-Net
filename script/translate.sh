#!/usr/bin/env bash

#PROJ=/home/evanyfgao/Distractor-Generation-RACE

export CUDA_VISIBLE_DEVICES=4
FULL_MODEL_NAME=$1

python3 -u ../translate.py \
    -model=data/model_step_300000.pt \
    -data=data/race_test_updated.json \
    -output=data/output \
    -share_vocab \
    -block_ngram_repeat=1 \
    -replace_unk \
    -batch_size=1 \
    -beam_size=10 \
    -n_best=10 \
    -gpuid=0 \
    -report_eval_every=500 \

