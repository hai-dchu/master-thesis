#!/usr/bin/env bash

python main_v1.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --job_name=train_stru3d \
               --epochs=3 \
               --dinov3_repo='/home/hai/master-thesis/RoomFormer/dinov3' \
               --dinov3_checkpoint='/home/hai/master-thesis/RoomFormer/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
