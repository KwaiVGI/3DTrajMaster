#!/bin/bash

export MODEL_PATH="/m2v_intern/fuxiao/CogVideo-release/weights/cogvideox-5b"    # Change it to CogVideoX-5B path
export TRANSFORMER_PATH=""                                                      # Resume from pretrained injector checkpoint
export LORA_PATH="/m2v_intern/fuxiao/CogVideo-release/weights/lora"             # Change it to pretrained lora path
export CACHE_PATH="~/.cache"
export DATASET_PATH="/ytech_m2v2_hdd/fuxiao/360Motion-Dataset"                  # Change it to 360-Motion Dataset path
export OUTPUT_PATH="injector"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,"

# if you are not using wth 8 gus, change `accelerate_config_machine_single_debug.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_cogvideox_injector.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --lora_path $LORA_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --finetune_init \
  --instance_data_root $DATASET_PATH \
  --validation_prompt "a woman with short black wavy hair, lean figure, a green and yellow plaid shirt, dark brown pants, and black suede shoes and a robotic gazelle with a sturdy aluminum frame, an agile build, articulated legs and curved, metallic horns are moving in the city" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --block_interval 2 \
  --seed 42 \
  --lora_scale 1.0 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 1000 \
  --checkpointing_steps 4000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb

  # --resume_from_checkpoint $TRANSFORMER_PATH \  