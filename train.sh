#!/bin/bash

pip install .
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=gs://bbahl/llama_debug
python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 16 \
  --do_train \
  --output_dir /home/$USER/tmp/test-clm \
  --overwrite_output_dir \
  --config_name llama-3-8b.json \
  --cache_dir /home/$USER/cache \
  --tokenizer_name meta-llama/Meta-Llama-3-8B \
  --block_size 8192 \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --fsdp "full_shard" \
  --fsdp_config llama-3-fsdp.json \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --flash_attention \
  --max_steps 10