# https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py
#
# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# regular:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import logging
import multiprocessing
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from typing import Optional
from dataclasses import dataclass, field


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


@dataclass
class ModelArguments(ModelConfig):
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    flash_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable PyTorch/XLA Pallas Flash Attention"
            )
        },
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    static: bool = field(
        default=True,
        metadata={
            "help": (
                "Make Mixtral's MoE static"
            )
        },
    )



if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelArguments))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
    )

    if model_config.config_name:
        config = AutoConfig.from_pretrained(model_config.config_name, **model_kwargs)

        # Pass the custom configs to the model config
        config.flash_attention = model_config.flash_attention
        config.static = model_config.static

        model = AutoModelForCausalLM.from_config(config)

        # Set the model dtype since we can no longer rely on USE_XLA_BF16.
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        print("model initialization finished")
        peft_config = get_peft_config(model_config)
        if peft_config is None:
            model_ref = AutoModelForCausalLM.from_config(config)
            if torch_dtype is not None:
                model_ref = model_ref.to(torch_dtype)
            print("model_ref initialization finished")
        else:
            model_ref = None
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        print("model initialization finished")

        config = model.config

        peft_config = get_peft_config(model_config)
        if peft_config is None:
            model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
            print("model_ref initialization finished")
        else:
            model_ref = None

    active_weight = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        # print(f"{name}: {numel} params")
        if 'block_sparse_moe.experts' in name:
            active_weight += numel / config.num_local_experts * config.num_experts_per_tok
        else:
            active_weight += numel
    print(f"Active weight: {active_weight:,}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=1,
        load_from_cache_file=False,
    )
    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split]

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
