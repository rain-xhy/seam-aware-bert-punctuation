#!/bin/bash
# Control trainings with crash-resume: each model is retried up to 5 times; train.py resumes from resume.pt.
PY="E:/anaconda/app/envs/yolov8/python.exe"
COMMON="--data_dir prepared_data_complete --bert_dir bert-base-chinese --epochs 3 --batch_size 1 --grad_accum 8 --seed 42 --save_every 4000"
run() {  # name, extra args...
  name=$1; shift
  for attempt in 1 2 3 4 5; do
    echo "[$(date '+%F %T')] $name attempt $attempt" >> logs/controls_queue.log
    $PY -X utf8 train.py $COMMON --out_dir checkpoints/$name "$@" >> logs/${name}_train.log 2>&1
    if grep -q "Training finished" logs/${name}_train.log; then
      echo "[$(date '+%F %T')] $name finished" >> logs/controls_queue.log; return 0
    fi
    sleep 30
  done
  echo "[$(date '+%F %T')] $name FAILED after 5 attempts" >> logs/controls_queue.log
}
run T3_noedge --max_len 400 --edge_aware_prob 0
run T4_classweight --max_len 400 --class_weights
run T5_centeronly --max_len 400 --center_only
run T2_fixed256 --min_len 256 --max_len 256
