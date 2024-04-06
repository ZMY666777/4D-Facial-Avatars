#!/bin/bash

rm -r renders
CUDA_VISIBLE_DEVICES=$1 python eval_transformed_rays.py --config ../../../datasets/nerface_dataset/person_4/person_4_config.yml --checkpoint logs/person_4/checkpoint$2.ckpt --savedir ./renders/person_4_rendered_frames

