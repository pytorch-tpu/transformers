# User Guide: Running HuggingFace Llama Training on Cloud TPUs


This user guide provides a concise overview of the essential steps required to run HuggingFace (HF) Llama training on Cloud TPUs.


## Environment Setup

The following setup assumes to run the training job with Llama 2 7B on GCE TPUs. Please follow corresponding TPU generation's user guide to setup the GCE TPUs. For GKE users, most of the commands below also apply.

### Setup Environment of Your TPUs
Please replace all your-* with your TPUs' information.
```
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
```

### HF Llama Environment Setup

Here both PyTorch and PyTorch/XLA nightly are used with our fork of HuggingFace.
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
# Step 1: install torch, torch-xla, libtpu, pallas
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl
pip3 install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip3 install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Step 2: install HF
git clone -b alanwaketan/flash_attention https://github.com/pytorch-tpu/transformers.git
cd transformers
pip3 install git+file://$PWD
pip3 install accelerate datasets evaluate scikit-learn huggingface-hub
'
```

The next step is to sign into HF such that you can get accesses to the tokenizer or model checkpoints. Please replace `your_token` with your HF token.
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
export PATH=$PATH:/home/$USER/.local/bin
huggingface-cli login --token your_token
'
```

The next step for HF setup is to copy your [Llama config](https://huggingface.co/meta-llama/Llama-2-7b-hf) into the TPU VM.
```
gcloud compute tpus tpu-vm scp Llama7B.json $TPU_NAME:~/config.json --worker all --project $PROJECT --zone=$ZONE
```

The last step for HF setup is to copy your fsdp_config.json into the TPU VM.
```
{
    "fsdp_transformer_layer_cls_to_wrap": [
        "LlamaDecoderLayer"
    ],
    "xla": true,
    "xla_fsdp_v2": true,
    "xla_fsdp_grad_ckpt": true
}

```
And the command to copy the config.
```
gcloud compute tpus tpu-vm scp fsdp_config.json $TPU_NAME:~/fsdp_config.json --worker all --project $PROJECT --zone=$ZONE
```

## Steps to Run HF Llama 2 7B
Following is the gcloud ssh command to run the training job from the host:
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
# Setup envs
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/

# Run
cd transformers
python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 8 \
  --do_train \
  --output_dir /tmp/test-clm \
  --overwrite_output_dir \
  --config_name ~/config.json \
  --cache_dir /tmp \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --block_size 4096 \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --fsdp "full_shard" \
  --fsdp_config ~/fsdp_config.json \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --flash_attention \
  --max_steps 5 \
'
```


### Environment Envs Explained



*   `PJRT_DEVICE`: Specify the XLA device.
*   `XLA_USE_SPMD`: Turn on GSPMD.
*   `XLA_IR_DEBUG`: Capture Python stack trace in Lazy IRs.
*   `XLA_HLO_DEBUG`: Capture Python stack trace in HLOs.
*   `PROFILE_EPOCH`: Specify which epoch to start taking the profile.
*   `PROFILE_STEP`: Specify which step to start taking the profile.
*   `PROFILE_DURATION_MS`: Specify how long the profiling will last.
*   `PROFILE_LOGDIR`: Specify where to put the profiling results.


### HF Mixtral Arguments Explained



*   `--flash_attention`: [bool] Enable Pallas FlashAttention. Default: False.
*   `--per_device_train_batch_size`: [int] Specify the global batch size. GSPMD treats the program as a singel device program.
*   `--num_train_epochs`: [int] Specify the total number of epochs. If the total steps is too large, try setting `--max_steps` instead to speed up the experiment.

## How to measure the step time?
A profile will be captured in `/tmp/home/`. Just use TensorBoard to open the profile and measure the step time from the "Trace View."
