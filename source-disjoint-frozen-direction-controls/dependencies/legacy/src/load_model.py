from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model_id: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: AutoConfig
    device: torch.device
    dtype: torch.dtype


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    normalized = dtype.lower().strip()
    if normalized == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        if device.type != "cuda":
            return torch.float32
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_device(device: str) -> torch.device:
    normalized = device.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def load_model_and_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    device: str = "auto",
    use_fast_tokenizer: bool = True,
    revision: Optional[str] = None,
    attn_implementation: Optional[str] = "eager",
    device_map: Optional[str] = None,
    max_memory: Optional[Mapping[object, object]] = None,
    offload_folder: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> LoadedModel:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    # Some older remote-code model implementations still expect rope_scaling["type"],
    # while newer config schemas expose rope_scaling["rope_type"] instead.
    rope_scaling = getattr(config, "rope_scaling", None)
    if trust_remote_code and getattr(config, "model_type", None) in {"phi3", "phi3small"} and isinstance(rope_scaling, dict):
        normalized_rope_scaling = dict(rope_scaling)
        if "type" not in normalized_rope_scaling and "rope_type" in normalized_rope_scaling:
            normalized_rope_scaling["type"] = normalized_rope_scaling["rope_type"]
        if normalized_rope_scaling.get("type") == "default":
            config.rope_scaling = None
        else:
            config.rope_scaling = normalized_rope_scaling

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
        revision=revision,
    )

    # Some official Mistral-family tokenizers need a post-load regex patch for
    # correct tokenization. transformers 5.3.0 warns about this family but does
    # not expose a reliable public loading flag in our environment, so patch the
    # backend tokenizer explicitly after loading.
    if getattr(config, "model_type", None) in {"mistral", "ministral", "mistral3", "voxtral", "pixtral"}:
        try:
            from transformers.tokenization_utils_tokenizers import TokenizersBackend

            tokenizer = TokenizersBackend._patch_mistral_regex(
                tokenizer,
                model_id,
                init_kwargs={},
                fix_mistral_regex=True,
            )
        except Exception:
            pass

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model_load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        "config": config,
    }
    if device_map is not None:
        model_load_kwargs["device_map"] = device_map
    if max_memory is not None:
        normalized_max_memory: Dict[object, object] = {}
        for key, value in max_memory.items():
            normalized_key = int(key) if isinstance(key, str) and key.isdigit() else key
            normalized_max_memory[normalized_key] = value
        model_load_kwargs["max_memory"] = normalized_max_memory
    if offload_folder is not None:
        model_load_kwargs["offload_folder"] = offload_folder
    if getattr(config, "model_type", None) == "phi3small":
        model_load_kwargs["low_cpu_mem_usage"] = False
    else:
        model_load_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    def _from_pretrained_with_dtype(**extra_kwargs):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=resolved_dtype,
                **model_load_kwargs,
                **extra_kwargs,
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=resolved_dtype,
                **model_load_kwargs,
                **extra_kwargs,
            )

    # Recent transformers defaults may pick SDPA, which suppresses attention weights.
    if attn_implementation is not None:
        try:
            model = _from_pretrained_with_dtype(attn_implementation=attn_implementation)
        except TypeError:
            model = _from_pretrained_with_dtype()
    else:
        model = _from_pretrained_with_dtype()

    dispatched = device_map is not None or hasattr(model, "hf_device_map")
    skip_resize = dispatched or getattr(config, "model_type", None) in {"gemma3", "gemma3_text"}
    # Gemma 3 tokenizers expose one extra multimodal-special token beyond the text LM
    # embedding table. Our text-only experiments never use that id, so avoid mutating the
    # checkpoint by resizing the embeddings just to match tokenizer length.
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer) and not skip_resize:
        model.resize_token_embeddings(len(tokenizer))

    if not dispatched:
        model.to(resolved_device)
    model.eval()

    return LoadedModel(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=resolved_device if not dispatched else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=resolved_dtype,
    )
