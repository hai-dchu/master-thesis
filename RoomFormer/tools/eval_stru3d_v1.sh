#!/usr/bin/env bash

echo "
Evalulating modified RoomFormer (v1)
In this version, dinov3 feature map are stacked
on top of the 2D projection map before passing
to the rest of RoomFormer. 
"

python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=output-v0/2026-07-01-10-27-13_train_stru3d/checkpoint.pth \
               --output_dir=eval_stru3d-retrain-vanilla \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
