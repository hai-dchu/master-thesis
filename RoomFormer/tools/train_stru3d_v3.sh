#!/usr/bin/env bash

python main_v3.py --dataset_name=stru3d \
               --dataset_root=data/stru3d_processed \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --job_name=train_dinorf_stru3d_1batch \
               --epochs=500 \
               --dinov3_repo='dinov3' \
               --dinov3_checkpoint='checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth' \
               --output_dir='output-v3' \
               --subset_length=10 \
               --batch_size=10 \
               --lr_dino_multilayer_proj=1e-3 \
               --wandb
