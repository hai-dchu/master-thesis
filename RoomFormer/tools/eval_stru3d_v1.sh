#!/usr/bin/env bash

echo "
Evalulating modified RoomFormer (v1)
In this version, dinov3 feature map are stacked
on top of the 2D projection map before passing
to the rest of RoomFormer. 
"

CHECKPOINT=$1

python eval_v1.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=$CHECKPOINT \
               --output_dir=eval_stru3d-dino-v1 \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --dinov3_repo=dinov3 \
               --dinov3_checkpoint=checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
