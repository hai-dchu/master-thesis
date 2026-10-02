#!/usr/bin/env bash

python main_v3.py --dataset_name=stru3d \
               --dataset_root=data/stru3d_processed \
               --num_queries=4000 \
               --num_polys=100 \
               --semantic_classes=19 \
               --job_name=drf_sem_rich \
               --epochs=500 \
               --batch_size=2 \
               --dinov3_repo='dinov3' \
               --dinov3_checkpoint='checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth' \
               --output_dir='output-v3' \
               --lr_dino_multilayer_proj=2e-3 \
               --wandb \
               --dino_bev_aggregation='random' \
               --pca='checkpoints/pca.pth' \
               --DINO_BEV