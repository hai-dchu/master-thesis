#!/usr/bin/env bash

echo "
Evalulating retrained vanilla RoomFormer (v0)
"

CHECKPOINT=$1

python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=$CHECKPOINT \
               --output_dir=eval_stru3d-retrain-vanilla \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
