#!/bin/bash


python decode.py \
  --model_lib models.hma_fusion \
  --model_name SimAM_HMAFusion \
  --decode_modal audiovideo \
  --checkpoint simam_hmafusion.pth \
  --test_audio_data misp_dataset/eval_far_v1 \
  ----test_video_data misp_dataset/eval_lips_video_v1