# syntax=docker/dockerfile:experimental
# Use torch_xla Python 3.10 as the base image
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_cxx11_20250113

ARG USE_LOCAL_WHEEL=false

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Add the Cloud Storage FUSE distribution URL as a package source
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-bullseye main" | tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install the Google Cloud SDK and GCS fuse
RUN apt-get update && apt-get install -y google-cloud-sdk git fuse gcsfuse && gcsfuse -v

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

WORKDIR /workspaces

RUN pip install jax==0.5.0.dev20250116 jaxlib==0.5.0.dev20250116 \
    -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html

COPY . /workspaces/transformers/
WORKDIR /workspaces/transformers/

RUN pip3 install git+file://$PWD accelerate datasets evaluate scikit-learn huggingface-hub

# Install torch and torch_xla from local wheels if USE_LOCAL_WHEEL and exists
# under local_dist directory. Note that you need to build the torch and
# torch_xla using the 
RUN if [ "$USE_LOCAL_WHEEL" = "true" ]; then \
        if [ -d "local_dist" ] && [ "$(find local_dist -name 'torch-*.whl' | wc -l)" -gt 0 ]; then \
            pip install local_dist/torch-*.whl; \
        else \
            echo "torch wheel not found in local_dist directory"; \
        fi; \
        if [ -d "local_dist" ] && [ "$(find local_dist -name 'torch_xla-*.whl' | wc -l)" -gt 0 ]; then \
            pip install local_dist/torch_xla-*.whl; \
        else \
            echo "torch_xla wheel not found in local_dist directory"; \
        fi; \
    fi
    
ENV LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=81920"
