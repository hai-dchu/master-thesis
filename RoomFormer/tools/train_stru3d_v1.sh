#!/usr/bin/env bash

python main_v1.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --job_name=train_stru3d \
               --epochs=3 \
               --lr_dinov3_head=0.001 \
               --dinov3_repo='dinov3' \
               --dinov3_checkpoint='checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
#               --wandb
