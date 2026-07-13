#!/usr/bin/env bash

echo "
Evalulating modified RoomFormer (v2)
In this version, dinov3 feature map are added
to the output of RoomFormer's backbone (ResNet50)
before passing to other modules.
"

CHECKPOINT=$1

python eval_v2.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=$CHECKPOINT \
               --output_dir=eval_stru3d-dino-v2 \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --dinov3_repo=dinov3 \
               --dinov3_checkpoint=checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
