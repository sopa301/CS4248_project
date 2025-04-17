#!/bin/bash

# Please define your own path here
huggingface_path=./huggingface-emoji/

# Multimodal models
CUDA_VISIBLE_DEVICES=7 python scripts/emote_multimodal.py --finetune 1 --model_name bart-large --portion 1 --seed 45 --hfpath $huggingface_path --use_images
