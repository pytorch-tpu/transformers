import logging
import unittest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import numpy as np
from transformers import AutoTokenizer, AutoConfig, MixtralForCausalLM

ATOL = 1e-3
RTOL = 1e-3

def setup_model(model_id, device, static=False, flash_attention=False, gmm=False, gmm_stack=False, seed=42):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(
        model_id,
        vocab_size=1024,
        torch_dtype=torch.bfloat16,
        num_hidden_layers=1,
        num_attention_heads=8,
        hidden_size=16,
        intermediate_size=64,
        num_local_experts=4,
    )
    config.static = static
    config.flash_attention = flash_attention
    config.gmm = gmm
    config.gmm_stack = gmm_stack
    torch.manual_seed(42)
    return MixtralForCausalLM(config).to(device)

def count_active_weights(model):
    active_weight = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            if 'block_sparse_moe.experts' in name:
                active_weight += numel / model.config.num_local_experts * model.config.num_experts_per_tok
            else:
                active_weight += numel
    return active_weight / 2**20

def compare_tensors(t1, t2, name=str):
    result = torch.allclose(t1, t2, atol=ATOL, rtol=RTOL)
    if result:
        return True
    else:
        print(f"{name=} {t1.shape=}")
        np.testing.assert_allclose(t1.cpu().numpy(), t2.cpu().numpy(), rtol=RTOL, atol=ATOL)
        return False

def run_test(input_size, dynamic_model, static_model, device):
    input_ids = torch.randint(0, 128, (2, input_size // 2)).to(device)
    # Create an attention mask with ones
    attention_mask = torch.ones_like(input_ids)

    # Define the desired sequence length and padding length
    seq_length = attention_mask.size(1)
    padding_length = 0  # Number of positions to pad

    # Ensure the padding length does not exceed the sequence length
    if padding_length > seq_length:
        raise ValueError("Padding length exceeds sequence length")

    # Apply padding to the mask
    attention_mask[:, seq_length - padding_length:] = 0
    # attention_mask = None
    
    dynamic_output = dynamic_model(input_ids, attention_mask=attention_mask).logits
    static_output = static_model(input_ids, attention_mask=attention_mask).logits

    print(dynamic_output.shape, static_output.shape)
    compare_tensors(dynamic_output.detach(), static_output.detach())

    dynamic_output.sum().backward()
    static_output.sum().backward()

    for (name, dynamic_param), static_param in zip(dynamic_model.named_parameters(), static_model.parameters()):
        compare_tensors(dynamic_param.grad, static_param.grad, name)

def main():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    device = xm.xla_device()

    dynamic_model = setup_model(model_id, device, static=False, flash_attention=False)
    print(f"Dynamic Model parameters: {dynamic_model.num_parameters()/2**20:.2f}M params")
    dynamic_model_active_weight = count_active_weights(dynamic_model)
    print(f"Dynamic Model Active weight: {dynamic_model_active_weight:.2f}M params")

    static_model = setup_model(model_id, device, static=True, flash_attention=True)
    print(f"Static Model parameters: {static_model.num_parameters()/2**20:.2f}M params")
    static_model_active_weight = count_active_weights(static_model)
    print(f"Static Model Active weight: {static_model_active_weight:.2f}M params")

    input_sizes = [256, 512]
    for input_size in input_sizes:
        run_test(input_size, dynamic_model, static_model, device)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # torch.set_default_dtype(torch.float32)
    # torch.manual_seed(42)
    # torch_xla._XLAC._xla_set_use_full_mat_mul_precision(use_full_mat_mul_precision=True)
    main()
