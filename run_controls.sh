#!/bin/bash
# Control trainings, same settings as T1 (batch 1, accum 8, seed 42), one switch changed each.
PY="E:/anaconda/app/envs/yolov8/python.exe"
COMMON="--data_dir prepared_data_complete --bert_dir bert-base-chinese --epochs 3 --batch_size 1 --grad_accum 8 --seed 42"
$PY -X utf8 train.py $COMMON --out_dir checkpoints/T2_fixed256 --min_len 256 --max_len 256 > logs/T2_fixed256_train.log 2>&1
$PY -X utf8 train.py $COMMON --out_dir checkpoints/T3_noedge --max_len 400 --edge_aware_prob 0 > logs/T3_noedge_train.log 2>&1
$PY -X utf8 train.py $COMMON --out_dir checkpoints/T4_classweight --max_len 400 --class_weights > logs/T4_classweight_train.log 2>&1
$PY -X utf8 train.py $COMMON --out_dir checkpoints/T5_centeronly --max_len 400 --center_only > logs/T5_centeronly_train.log 2>&1
