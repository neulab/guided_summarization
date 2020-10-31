#!/bin/sh     
BERT_DATA_PATH=$1
MODEL_PATH=$2
RESULT_PATH=$3

python -W ignore z_train.py \
    -task abs \
    -mode test \
    -batch_size 3000 \
    -test_batch_size 1500 \
    -bert_data_path $BERT_DATA_PATH \
    -log_file ../logs/test.logs \
    -test_from $MODEL_PATH \
    -sep_optim true \
    -use_interval true \
    -visible_gpus 0 \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length 50 \
    -result_path $RESULT_PATH
