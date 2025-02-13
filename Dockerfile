FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_cxx11_20250211

# Set the working directory
WORKDIR /workspace

RUN pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Clone the PyTorch TPU diffusers repository
RUN git clone -b flash_attention https://github.com/pytorch-tpu/transformers.git

# Set the working directory to the transformers repository
WORKDIR /workspace/transformers

# Install required Python packages for the example
RUN pip3 install .
RUN pip3 install accelerate datasets evaluate scikit-learn huggingface-hub protobuf

WORKDIR /workspace/

# Set the default command (optional, for example, to run Python)
CMD ["/bin/bash"]