import json
import numpy as np
import torch
from collections import defaultdict


ARCH_KEYS = ["Qwen2", "Qwen3", "Llama", "Mistral", "zephyr", "Phi3", "Lfm2"]


def get_layers(model):
    if any(x in str(model) for x in ARCH_KEYS):
        return list(model.model.layers)
    elif "Gemma3" in str(model):
        return list(model.model.language_model.layers)
    else:
        assert 0, f"get_layers not implemented for model {str(type(model))}"


def get_embed_tokens(model):
    if any(x in str(model) for x in ARCH_KEYS):
        return model.model.embed_tokens
    elif "Gemma3" in str(model):
        return model.model.language_model.embed_tokens
    else:
        assert 0, f"get_embed_tokens not implemented for model {str(type(model))}"


def gen_hook_func_steering(layer_idx, hook_dict):
    def hook_steering(module, inputs):
        if layer_idx not in hook_dict:
            return None
        hidden_states = inputs[0]
        if hidden_states.shape[1] <= 1:
            return None
        vec = hook_dict[layer_idx]
        if not torch.is_tensor(vec):
            vec = torch.tensor(vec)
        vec = vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[:, -2] += vec
        return (hidden_states,) + inputs[1:]
    return hook_steering


def init_model_hook_steering(model):
    hook_dict = {}
    for k, layer in enumerate(get_layers(model)):
        layer.register_forward_pre_hook(gen_hook_func_steering(layer_idx=k, hook_dict=hook_dict))
    return hook_dict


def load_steering_vec(steering_vec_file, layer_idx, scale):
    vec_dict = json.load(open(steering_vec_file, "r"))
    key = layer_idx if layer_idx in vec_dict else str(layer_idx)
    vec = np.array(vec_dict[key], dtype=np.float32) * scale
    return torch.tensor(vec)


def gen_hook_func_head_ablation(head_idx):
    def hook_head_ablation(module, inp, out):
        if len(out) < 2 or out[1] is None:
            return out
        attn_weights = out[1].clone()
        attn_weights[:, head_idx, :, :] = 0.0
        return (out[0], attn_weights) + out[2:]
    return hook_head_ablation


def register_head_ablation(model, layer_idx, head_idx):
    attn = get_layers(model)[layer_idx].self_attn
    if hasattr(attn, "config"):
        attn.config._attn_implementation = "eager"
    return attn.register_forward_hook(gen_hook_func_head_ablation(head_idx))
