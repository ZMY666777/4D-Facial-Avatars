#!/bin/bash

rm -r renders/
CUDA_VISIBLE_DEVICES=$1 python eval_transformed_rays.py --config ../../../datasets/nerface_dataset/person_1/person_1_config.yml --checkpoint logs/person_1/checkpoint$2.ckpt --savedir ./renders/person_1_rendered_frames
