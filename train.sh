#!/bin/bash

pip install .
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=gs://bbahl/mixtral_expert_parallel
export USE_EXPERT_PARALLELISM=0
python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 8 \
  --do_train \
  --output_dir /tmp/test-clm \
  --overwrite_output_dir \
  --config_name mixtral_2_experts.json \
  --cache_dir /tmp \
  --tokenizer_name mistralai/Mixtral-8x7B-v0.1 \
  --block_size 1024 \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --flash_attention \
  --num_train_epochs 1 \
  --static \
  --max_steps 15 \
  --fsdp "full_shard" \
  --fsdp_config fsdp_config.json