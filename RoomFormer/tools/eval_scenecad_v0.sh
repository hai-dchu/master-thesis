#!/usr/bin/env bash

CHECKPOINT=$1

python eval.py --dataset_name=scenecad \
               --dataset_root=data/scenecad \
               --eval_set=val \
               --checkpoint=$CHECKPOINT \
               --output_dir=eval_scenecad \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
