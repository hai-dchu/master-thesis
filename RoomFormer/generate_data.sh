#!/usr/bin/env bash

# generate point cloud and reduced npy version
# cd /home/hai/master-thesis/RoomFormer
# python data_preprocess/stru3d/generate_data.py --normal

# generate annotations
cd data_preprocess/stru3d/
python generate_annotations.py --data_root /home/hai/master-thesis/RoomFormer/data/Structured3D