# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
# isort: off
from .integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
import torch_xla.distributed.parallel_loader as pl
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    logging,
    strtobool,
)
from .utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

if is_safetensors_available():
    import safetensors.torch

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        GradientAccumulationPlugin,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration

if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)
SINGLE_SLICE=os.environ.get('SINGLE_SLICE', None)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.

            </Tip>

        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.
        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs to the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values.
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """

    # Those are used as methods of the Trainer in examples.
    from .trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        
        self.is_model_parallel = False

        # True
        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]
        if len(args.fsdp) > 0:
            if not args.fsdp_config["xla"] and args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise ValueError("Using fsdp only works in distributed training.")

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = None
        if (
            self.is_fsdp_xla_enabled
        ):
            self.place_model_on_device = False

        default_collator = (
            DataCollatorWithPadding(tokenizer)
            if tokenizer is not None and isinstance(tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
            else default_data_collator
        )
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_xla_available() and self.optimizer is not None:
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )
        if (self.is_fsdp_xla_enabled) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self._signature_columns = None
        self.label_smoother = None

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)
        if self.is_fsdp_xla_v2_enabled:
            if not IS_XLA_FSDPV2_POST_2_2:
                raise ValueError("FSDPv2 requires `torch_xla` 2.2 or higher.")
            # Prepare the SPMD mesh that is going to be used by the data loader and the FSDPv2 wrapper.
            # Tensor axis is just a placeholder where it will not be used in FSDPv2.
        num_devices = xr.global_runtime_device_count()

        if SINGLE_SLICE:
            mesh_shape = (num_devices, 1)
            device_ids = np.array(range(num_devices))
            # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
            mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'tensor'))
            xs.set_global_mesh(mesh)
        else:
            dcn_axis = 2
            ici_mesh_shape = (1, num_devices // dcn_axis)
            dcn_mesh_shape = (dcn_axis, 1)
            mesh = xs.HybridMesh(ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape, axis_names=('dcn', 'fsdp'))
            xs.set_global_mesh(mesh)
        # xs.set_global_mesh(xs.Mesh(np.array(range(num_devices)), (num_devices, 1), axis_names=("fsdp", "tensor")))

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def _move_model_to_device(self, model, device):
        model = model.to(device)
        logger.warning("Moving model to device")
        # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        
        torch_dataloader = DataLoader(train_dataset, **dataloader_params)
        device = xm.xla_device()
        if SINGLE_SLICE:
            mp_device_loader = pl.MpDeviceLoader(
                torch_dataloader,
                device,
                input_sharding=xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None)),
            )
        else:
            mp_device_loader = pl.MpDeviceLoader(
                torch_dataloader,
                device,
                input_sharding=xs.ShardingSpec(xs.get_global_mesh(), (("dcn", "fsdp"), None)),
            )

        return mp_device_loader

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def get_num_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_learning_rates(self):
        """
        Returns the learning rate of each parameter from self.optimizer.
        """
        if self.optimizer is None:
            raise ValueError("Trainer optimizer is None, please make sure you have setup the optimizer before.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_optimizer_group(self, param: Optional[Union[str, torch.nn.parameter.Parameter]] = None):
        """
        Returns optimizer group for a parameter if given, else returns all optimizer groups for params.

        Args:
            param (`str` or `torch.nn.parameter.Parameter`, *optional*):
                The parameter for which optimizer group needs to be returned.
        """
        if self.optimizer is None:
            raise ValueError("Trainer optimizer is None, please make sure you have setup the optimizer before.")
        if param is not None:
            for group in self.optimizer.param_groups:
                if param in group["params"]:
                    return group
        return [group["params"] for group in self.optimizer.param_groups]

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from .optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        elif args.optim == OptimizerNames.RMSPROP:
            optimizer_cls = torch.optim.RMSprop
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            for step, batch in enumerate(train_dl):
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens

    def call_model_init(self, trial=None):
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model


    def _wrap_model(self, model, training=True, dataloader=None):
        if self.is_fsdp_xla_enabled:
            try:
                from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
                from torch_xla.distributed.fsdp import checkpoint_module
                from torch_xla.distributed.fsdp.wrap import (
                    size_based_auto_wrap_policy,
                    transformer_auto_wrap_policy,
                )

                if self.is_fsdp_xla_v2_enabled:
                    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
                        SpmdFullyShardedDataParallel as FSDPv2,
                    )
            except ImportError:
                raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
            auto_wrap_policy = None
            auto_wrapper_callable = None
            default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
            fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get(
                "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
            )

            if self.args.fsdp_config["min_num_params"] > 0:
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["min_num_params"]
                )
            elif fsdp_transformer_layer_cls_to_wrap is not None:
                transformer_cls_to_wrap = set()
                for layer_class in fsdp_transformer_layer_cls_to_wrap:
                    transformer_cls = get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)

                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
            if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
                if model.config.use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                    )
                    model.config.use_cache = False
                # Apply gradient checkpointing to auto-wrapped sub-modules if specified
                def auto_wrapper_callable(m, *args, **kwargs):
                    target_cls = FSDP if not self.is_fsdp_xla_v2_enabled else FSDPv2
                    return target_cls(checkpoint_module(m), *args, **kwargs)

            # Wrap the base model with an outer FSDP wrapper
            if self.is_fsdp_xla_v2_enabled:

                def shard_output(output, mesh):
                    from .modeling_outputs import CausalLMOutputWithPast

                    real_output = None
                    if isinstance(output, torch.Tensor):
                        real_output = output
                    elif isinstance(output, tuple):
                        real_output = output[0]
                    elif isinstance(output, CausalLMOutputWithPast):
                        real_output = output.logits

                    if real_output is None:
                        raise ValueError("Something went wrong, the output of the model shouldn't be `None`")
                    if SINGLE_SLICE:
                        xs.mark_sharding(real_output, mesh, ("fsdp", None, None))
                    else:
                        xs.mark_sharding(real_output, mesh, (("dcn", "fsdp"), None, None))

                self.model = model = FSDPv2(
                    model,
                    shard_output=shard_output,
                    auto_wrap_policy=auto_wrap_policy,
                    auto_wrapper_callable=auto_wrapper_callable,
                )
            else:
                fsdp_kwargs = self.args.xla_fsdp_config
                self.model = model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap_policy,
                    auto_wrapper_callable=auto_wrapper_callable,
                    reshard_after_forward=False,
                    **fsdp_kwargs,
                )
        return model

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        args = self.args

        self.is_in_train = True
        # This might change the seed so needs to run first.
        self._train_batch_size = self.args.train_batch_size
        
        return self._inner_training_loop(
            batch_size=self._train_batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # if self.is_fsdp_xla_v2_enabled:
        #     pass
        #     train_dataloader = tpu_spmd_dataloader(train_dataloader)
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            assert args.max_steps > 0
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
        ## DEBUG: This is true when FSDP is being used.
        assert self.is_fsdp_xla_enabled
        assert not self._created_lr_scheduler

        self.state = TrainerState()
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False
        self.model = self.model.to(self.accelerator.device)
        # if use_accelerator_prepare:
        #     self.model = self.accelerator.prepare(self.model)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            # This is interesting. Moves optimizer state dict to device and syncs gradients manually?
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        server = xp.start_server(9012)
        logger.info(f'Profiling server started: {str(server)}')
        profile_step = int(os.environ.get('PROFILE_STEP', -1))
        profile_epoch = int(os.environ.get('PROFILE_EPOCH', -1))
        profile_duration = int(os.environ.get('PROFILE_DURATION_MS', 20000))
        profile_logdir = os.environ.get('PROFILE_LOGDIR', None)
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                tr_loss += tr_loss_step

                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    # deepspeed does its own clipping
                    ## Calls all reduce on gradients and clips to the max value.
                    _grad_norm = self.accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )
                    grad_norm = _grad_norm

                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                if step == profile_step and epoch == profile_epoch:
                    # Wait until device execution catches up to tracing before triggering the profile. This will
                    # interrupt training slightly on the hosts which are capturing, but by waiting after tracing
                    # for the step, the interruption will be minimal.
                    xm.wait_device_ops()
                    import tempfile
                    xp.trace_detached('127.0.0.1:9012', profile_logdir or tempfile.mkdtemp(), profile_duration or 20000)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            if self.control.should_training_stop:
                break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_steps=self.state.max_steps,
        )
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        loss = self.compute_loss(model, inputs)

        # self.accelerator.backward(loss)
        loss.backward()

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        loss = outputs["loss"]
        return loss

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        return self.args.process_index == 0

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {}
       
        grad_acc_kwargs["num_steps"] = self.args.gradient_accumulation_steps

        grad_acc_kwargs["sync_with_dataloader"] = False

        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_config = self.args.accelerator_config.to_dict()

        if is_accelerate_available("0.28.0"):
            dataloader_config = DataLoaderConfiguration(
                split_batches=accelerator_config.pop("split_batches"),
                dispatch_batches=accelerator_config.pop("dispatch_batches"),
                even_batches=accelerator_config.pop("even_batches"),
                use_seedable_sampler=accelerator_config.pop("use_seedable_sampler"),
            )
        # this would have been updated above, no need for it anymore
        accelerator_config.pop("gradient_accumulation_kwargs")

        args = {
            "gradient_accumulation_plugin": gradient_accumulation_plugin,
        }
        if is_accelerate_available("0.28.0"):
            args["dataloader_config"] = dataloader_config
        else:
            args.update(accelerator_config)

        # create accelerator object
        self.accelerator = Accelerator(**args)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics