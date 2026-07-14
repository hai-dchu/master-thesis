#!/usr/bin/env bash

CHECKPOINT=$1

python eval_v2.py --dataset_name=scenecad \
               --dataset_root=data/scenecad \
               --eval_set=val \
               --checkpoint=$CHECKPOINT \
               --output_dir=eval_scenecad \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --dinov3_repo=dinov3 \
               --dinov3_checkpoint=checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
