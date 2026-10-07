from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    tokenizer: Any
    config: Any
    device: torch.device
    dtype: torch.dtype
    original_num_hidden_layers: int
    active_num_hidden_layers: int


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable")
    return requested


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    normalized = dtype.lower()
    if normalized == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        selected = choices[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {dtype}") from error
    if device.type != "cuda" and selected in {torch.bfloat16, torch.float16}:
        return torch.float32
    return selected


def truncate_decoder_layers(model: Any, keep_layers: int) -> int:
    """Keep a causal decoder prefix and return its original layer count.

    Value-write scores from a writer window only depend on layers through the
    end of that window. Truncating later causal layers therefore preserves the
    selector exactly while avoiding their forward-pass cost.
    """

    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose decoder layers as model.layers")
    original = len(layers)
    if not 1 <= keep_layers <= original:
        raise ValueError(
            f"keep_layers must be in [1, {original}], received {keep_layers}"
        )
    if keep_layers < original:
        decoder.layers = nn.ModuleList(list(layers[:keep_layers]))
    configs = {
        id(model.config): model.config,
        id(getattr(decoder, "config", None)): getattr(decoder, "config", None),
    }
    for config in configs.values():
        if config is not None and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = keep_layers
    return original


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
    revision: str | None = None,
    truncate_to_layers: int | None = None,
) -> LoadedModel:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        revision=revision,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    common: dict[str, Any] = {
        "config": config,
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        common["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=resolved_dtype, **common
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=resolved_dtype, **common
        )
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    original_num_hidden_layers = len(model.model.layers)
    if truncate_to_layers is not None:
        original_num_hidden_layers = truncate_decoder_layers(
            model, truncate_to_layers
        )
    active_num_hidden_layers = len(model.model.layers)
    model.to(resolved_device)
    model.eval()
    return LoadedModel(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=resolved_device,
        dtype=resolved_dtype,
        original_num_hidden_layers=original_num_hidden_layers,
        active_num_hidden_layers=active_num_hidden_layers,
    )


@torch.inference_mode()
def generate_answer(
    loaded: LoadedModel, prompt: str, max_new_tokens: int
) -> tuple[str, str, int, int]:
    from .prompting import short_answer

    encoded = loaded.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
        loaded.device
    )
    output = loaded.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=loaded.tokenizer.eos_token_id,
        use_cache=True,
    )
    generated_ids = output[0, input_ids.shape[1] :]
    raw = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return raw, short_answer(raw), int(input_ids.shape[1]), int(generated_ids.shape[0])


@torch.inference_mode()
def generate_answer_with_timing(
    loaded: LoadedModel, prompt: str, max_new_tokens: int
) -> tuple[str, str, int, int, float, float, float, float]:
    """Generate greedily and split reader latency into TTFT and decode time.

    TTFT begins before CPU tokenization and ends when the first generated token
    is available after a device synchronization.  Decode time is the remaining
    generation time.  The returned total also includes tokenization.
    """

    from transformers.generation.streamers import BaseStreamer

    from .prompting import short_answer
    from .timing import synchronize

    request_started = time.perf_counter()
    encoded = loaded.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
        loaded.device
    )
    tokenization_seconds = time.perf_counter() - request_started

    class TimingStreamer(BaseStreamer):
        def __init__(self) -> None:
            self.calls = 0
            self.first_token_at: float | None = None

        def put(self, value: torch.Tensor) -> None:
            self.calls += 1
            if self.calls == 1:
                return
            if self.first_token_at is None:
                synchronize(loaded.device)
                self.first_token_at = time.perf_counter()

        def end(self) -> None:
            return None

    synchronize(loaded.device)
    streamer = TimingStreamer()
    output = loaded.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=loaded.tokenizer.eos_token_id,
        use_cache=True,
        streamer=streamer,
    )
    synchronize(loaded.device)
    finished = time.perf_counter()
    generated_ids = output[0, input_ids.shape[1] :]
    raw = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    total_seconds = finished - request_started
    ttft_seconds = (
        total_seconds
        if streamer.first_token_at is None
        else streamer.first_token_at - request_started
    )
    decode_seconds = max(0.0, total_seconds - ttft_seconds)
    return (
        raw,
        short_answer(raw),
        int(input_ids.shape[1]),
        int(generated_ids.shape[0]),
        ttft_seconds,
        decode_seconds,
        total_seconds,
        tokenization_seconds,
    )
