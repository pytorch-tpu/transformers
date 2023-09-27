# User Guide: Running HuggingFace Llama 2 Training on v4 and v5e


This user guide provides a concise overview of the essential steps required to run HuggingFace (HF) Llama 2 training on both v4 and v5e. It first demonstrates how to do so on barebone TPU VMs, and then shares a Dockerfile for those who would prefer to run it in a container.


## Environment Setup

The following setup assumes to run the training job with Llama 2 7B.


### Cloud TPU VM Creation
```
export TPU_NAME=your-tpu-name
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-32
export RUNTIME_VERSION=tpu-ubuntu2204-base
export PROJECT=your-project

gcloud alpha compute tpus queued-resources create $TPU_NAME
  \ --node-id=$TPU_NAME
  \ --zone=$ZONE
  \ --project=$PROJECT
  \ --accelerator-type=$ACCELERATOR_TYPE
  \ --runtime-version=$RUNTIME_VERSION
```

Change `{TPU_NAME, ZONE, PROJECT, RUNTIME_VERSION}` as needed. For v5e, use the `v2-alpha-tpuv5-lite` runtime version in a supported project and zone. And use the following command to query the status of the creation request:
```
gcloud alpha compute tpus queued-resources describe ${TPU_NAME}  --project ${PROJECT} --zone ${ZONE}
```

Once the QueuedResource’s status becomes ACTIVE, the TPU VMs are provisioned and ready to run your training workload. See the [official guide](https://cloud.google.com/tpu/docs/queued-resources) for more information on a QueuedResource’s life cycle.


### HF Llama 2 Environment Setup

Here PyTorch 2.1 and PyTorch/XLA 2.1 are used with our fork of HuggingFace. At the time where this doc was created, the 2.1 releases were not officially out yet, and therefore, pre-releases links are used.
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
# Step 1: install torch, torch-xla, libtpu
pip3 install torch~=2.1.0 torch_xla[tpu]~=2.1.0rc5 -f https://storage.googleapis.com/libtpu-releases/index.html

# Step 2: install HF
git clone -b llama2-google-next-training https://github.com/pytorch-tpu/transformers.git
cd transformers
sudo pip3 install -e . --user
pip3 install datasets accelerate evaluate scikit-learn
'
```

The last step for HF setup is to copy your Llama 2 config into the TPU VM.
```
gcloud compute tpus tpu-vm scp llama_2_7B.config $TPU_NAME:~/config.json --worker all --project $PROJECT --zone=$ZONE
```


## Steps to Run HF Llama 2
Following is the gcloud ssh command to run the training job from the host:
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
# Setup envs
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/

# Run
cd transformers
python examples/pytorch/language-modeling/run_clm.py \
  --tokenizer_name hf-internal-testing/llama-tokenizer \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --do_train \
  --output_dir /tmp/output \
  --overwrite_output_dir \
  --config_name ~/config.json \
  --save_strategy no \
  --logging_strategy no \
  --remove_unused_columns no \
  --optim adafactor \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --block_size 1024 \
  --spmd_2d_sharding 1 \
  --spmd_grad_chkpt
'
```


### Environment Envs Explained



*   `PJRT_DEVICE`: Specify the XLA device.
*   `XLA_USE_BF16`: Force to use bfloat16 as default dtype.
*   `XLA_IR_DEBUG`: Capture Python stack trace in Lazy IRs.
*   `XLA_HLO_DEBUG`: Capture Python stack trace in HLOs.
*   `PROFILE_EPOCH`: Specify which epoch to start taking the profile.
*   `PROFILE_STEP`: Specify which step to start taking the profile.
*   `PROFILE_DURATION_MS`: Specify how long the profiling will last.
*   `PROFILE_LOGDIR`: Specify where to put the profiling results.


### HF SPMD Arguments Explained



*   `--spmd_grad_chkpt`: [bool] Apply gradient checkpointing to the transformer blocks. Default: False.
*   `--spmd_2d_sharding <model_dim>`: [int] Specify the size of the model axis in the device mesh for 2D sharding. This flag is exclusive with spmd\_fsdp\_sharding. Default: 0 (indicating no 2D sharding)
*   `--spmd_fsdp_sharding`: [bool] Apply FSDP sharding to model parameters using the SPMD sharding API. This flag is exclusive with spmd\_2d\_sharding. Default: False
*   `--spmd_defer_init`: [bool] Defer model parameter initialization until sharding is applied. This will alleviate host-side memory pressure for larger variants of the model. Default: False
*   `--spmd_dcn_parallelism <slice_count>`: [int] Specify the number of slices to run data parallel. This controls the `dcn` axis of the device mesh. Default: 1
*   `--spmd_debug`: [bool] Print verbose debug logs related to SPMD. Default: False


## Steps to Run HF Llama 2 in Docker

To run using Docker, you can bake the above commands into an image that is shared across your worker VMs or even used in GKE. To follow this guide, you will need write access to a docker repo that your TPU VMs have read access to. This can be achieved using [Artifact Registry](https://cloud.google.com/artifact-registry).

The following Dockerfile will build an image which runs a Llama 2 training workload:
```
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_tpuvm

# Allow overriding some training parameters at build time
ARG spmd_sharding_flag="--spmd_2d_sharding 2"
ARG train_config=config.json
ARG global_batch_size=256
ARG libtpu_init_args=""

# Clone and install the SPMD-enabled fork of HF transformers
RUN git clone -b llama2-google-next-training https://github.com/pytorch-tpu/transformers.git && \
    pip install git+file:///transformers datasets accelerate evaluate scikit-learn

# Copy the config file from the build context
COPY ${train_config} /config.json

# Copy relevant args to environment variables for use in CMD
ENV SPMD_SHARDING_FLAG="${spmd_sharding_flag}"
ENV GLOBAL_BATCH_SIZE="${global_batch_size}"
ENV LIBTPU_INIT_ARGS="${libtpu_init_args}"

# Run the training using the copied config file and specified sharding strategy
CMD python -u \
  /transformers/examples/pytorch/language-modeling/run_clm.py \
  --tokenizer_name hf-internal-testing/llama-tokenizer \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size ${GLOBAL_BATCH_SIZE} \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 3 \
  --do_train \
  --output_dir /tmp/output \
  --overwrite_output_dir \
  --config_name /config.json \
  --save_strategy no --logging_strategy no \
  --remove_unused_columns no \
  --optim adafactor \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --spmd_grad_chkpt \
  --spmd_defer_init \
  ${SPMD_SHARDING_FLAG}
```

The following build arguments can be used to customize the build:
*   `spmd_sharding_flag`: Used to configure the sharding strategy used. Valid values are `--spmd_fsdp_sharding` and` --spmd_2d_sharding &lt;model_dim>`, where `model_dim` is the size of the model axis in the device mesh. The default is `--spmd_2d_sharding 2`.
*   `train_config`: Specify the HuggingFace training config to copy from the build context into the image. The default value is `config.json`, which may not exist in your directory.
*   `global_batch_size`: The global batch size to use. Note that this value is supplied to the `per_device_train_batch_size` flag, since currently HuggingFace treats SPMD as a single-device program. This will change in future releases.
*   `libtpu_init_args`: This is optional and can be used to pass XLA flags to the compiler.

To build and push the image, copy the above Dockerfile into a directory containing the desired training config, and use the following commands to build and push the image:
```
export DOCKER_IMAGE_TAG=<your_image_and_tag>

# Here, we override the spmd_sharding_flag to use a model dimension of
# size 4 with 2D sharding. In this example, the 70B config is in the
# file ./llama2_70B.json
docker build -t ${DOCKER_IMAGE_TAG} \
  --build-arg spmd_sharding_flag='--spmd_2d_sharding 4' \
  --build-arg global_batch_size=128 \
  --build-arg train_config=llama2_70B.json .

# Push the image to a repository. The repo must be accessible by the
# TPU VM's service account to pull the image.
docker push ${DOCKER_IMAGE_TAG}
```

To run the image on a TPU VM provisioned in GCE, use the following ssh command. Note that every worker’s output will be streamed to your console, which can be quite noisy for larger slices:
```
# Run the training workload. Note that the container must be privileged and use the
# host network.
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --worker all --command 'sudo docker run --rm --privileged --net host ${DOCKER_IMAGE_TAG}'
```

To use in GKE, create a Pod using your image in a TPU-VM-enabled cluster. See the [TPUs in GKE](https://cloud.google.com/tpu/docs/tpus-in-gke) guide for more details.
