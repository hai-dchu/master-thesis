#!/usr/bin/env bash

echo "
Evalulating vanilla RoomFormer on self-compiled dataset (v3)
"

CHECKPOINT=checkpoints/roomformer_stru3d.pth

python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d_processed \
               --eval_set=test \
               --checkpoint=$CHECKPOINT \
               --output_dir=eval_stru3d-vanilla-v3-dataset \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
