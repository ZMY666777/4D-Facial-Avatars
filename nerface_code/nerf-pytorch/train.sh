#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train_tensorf.py --config ../../../datasets/nerface_dataset/person_1/person_1_config.yml
