from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .model_loading import LoadedModel
from .prompting import render_prompt, token_positions


@dataclass
class DocumentFeatures:
    prompt: str
    document_ids: list[str]
    vectors: np.ndarray
    attention_mass: np.ndarray
    token_lengths: np.ndarray
    prompt_tokens: int


@dataclass(frozen=True)
class FrozenDirection:
    mean: np.ndarray
    component: np.ndarray
    component_index: int
    sign: int


def load_frozen_direction(path: Path, expected_hidden_size: int) -> FrozenDirection:
    with np.load(path) as data:
        direction = FrozenDirection(
            mean=data["mean"].astype(np.float64),
            component=data["component"].astype(np.float64),
            component_index=int(data["component_index"]),
            sign=int(data["sign"]),
        )
    expected_shape = (expected_hidden_size,)
    if direction.mean.shape != expected_shape or direction.component.shape != expected_shape:
        raise ValueError(
            f"Direction shape mismatch: expected {expected_shape}, got "
            f"{direction.mean.shape} and {direction.component.shape}"
        )
    if direction.sign not in {-1, 1}:
        raise ValueError(f"Direction sign must be -1 or 1, got {direction.sign}")
    return direction


class DocumentFeatureExtractor:
    """Collect source-to-final-position attention value writes."""

    def __init__(self, model: Any, writer_window: Sequence[int]):
        self.model = model
        self.writer_window = tuple(int(index) for index in writer_window)
        self.layers = list(model.model.layers)
        if not self.writer_window:
            raise ValueError("writer_window cannot be empty")
        if min(self.writer_window) < 0 or max(self.writer_window) >= len(self.layers):
            raise ValueError(
                f"writer_window {self.writer_window} is invalid for {len(self.layers)} layers"
            )
        self.value_cache: dict[int, torch.Tensor] = {}
        self.handles = [
            self.layers[layer_index].self_attn.v_proj.register_forward_hook(
                self._make_hook(layer_index)
            )
            for layer_index in self.writer_window
        ]

    def _make_hook(self, layer_index: int):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.value_cache[layer_index] = output.detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.value_cache.clear()

    @torch.inference_mode()
    def collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.value_cache.clear()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.attentions is None:
            raise RuntimeError(
                "The reader did not return attention weights. Load it with eager attention."
            )
        config = self.model.config
        n_query_heads = int(config.num_attention_heads)
        n_key_value_heads = int(
            getattr(config, "num_key_value_heads", n_query_heads)
        )
        head_dim = int(
            getattr(config, "head_dim", config.hidden_size // n_query_heads)
        )
        if n_query_heads % n_key_value_heads:
            raise RuntimeError("Query heads are not divisible by key/value heads")
        repeat_groups = n_query_heads // n_key_value_heads
        target_position = input_ids.shape[1] - 1
        device = input_ids.device
        document_vectors = torch.zeros(
            (len(positions), int(config.hidden_size)),
            device=device,
            dtype=torch.float32,
        )
        attention_mass = torch.zeros(
            len(positions), device=device, dtype=torch.float32
        )
        position_tensors = [
            torch.as_tensor(item, device=device, dtype=torch.long) for item in positions
        ]

        for layer_index in self.writer_window:
            if layer_index not in self.value_cache:
                raise RuntimeError(f"Missing v_proj output for layer {layer_index}")
            raw_values = self.value_cache[layer_index][0]
            sequence_length = raw_values.shape[0]
            values = raw_values.view(sequence_length, n_key_value_heads, head_dim)
            values = values.repeat_interleave(repeat_groups, dim=1)
            attention = outputs.attentions[layer_index]
            if attention is None:
                raise RuntimeError(f"Missing attention tensor for layer {layer_index}")
            attention = attention[0, :, target_position, :].float()
            output_weight = self.layers[layer_index].self_attn.o_proj.weight
            aggregated: list[torch.Tensor] = []
            for document_index, document_positions in enumerate(position_tensors):
                document_positions = document_positions[
                    document_positions < sequence_length
                ]
                if document_positions.numel() == 0:
                    aggregated.append(
                        torch.zeros(
                            n_query_heads * head_dim,
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    continue
                selected_values = values.index_select(0, document_positions)
                selected_attention = attention.index_select(1, document_positions).T
                weighted = selected_values.float() * selected_attention.unsqueeze(-1)
                aggregated.append(weighted.sum(dim=0).reshape(-1))
                attention_mass[document_index] += selected_attention.sum()
            head_space = torch.stack(aggregated).float()
            document_vectors += torch.matmul(head_space, output_weight.float().T)

        token_lengths = np.asarray([len(item) for item in positions], dtype=np.int64)
        result = (
            document_vectors.cpu().numpy(),
            attention_mass.cpu().numpy(),
            token_lengths,
        )
        del outputs, document_vectors, attention_mass
        self.value_cache.clear()
        return result


class FinalQueryDocumentFeatureExtractor:
    """Extract the same final-query value writes without dense attention maps.

    The normal eager implementation materializes every query-by-key attention
    matrix.  Long-context selection only needs the final query row.  This
    extractor runs the decoder with its memory-efficient attention backend,
    captures pre-RoPE Q/K/V projections at the frozen writer layers, and
    reconstructs that one row exactly after the forward pass.

    The implementation follows the Llama-family rotary/GQA attention used by
    the frozen 3B scout.  It deliberately rejects unsupported architectures.
    """

    def __init__(self, model: Any, writer_window: Sequence[int]):
        self.model = model
        self.writer_window = tuple(int(index) for index in writer_window)
        self.layers = list(model.model.layers)
        if not self.writer_window:
            raise ValueError("writer_window cannot be empty")
        if min(self.writer_window) < 0 or max(self.writer_window) >= len(self.layers):
            raise ValueError(
                f"writer_window {self.writer_window} is invalid for {len(self.layers)} layers"
            )
        self.cache: dict[int, dict[str, torch.Tensor]] = {
            index: {} for index in self.writer_window
        }
        self.handles = []
        for layer_index in self.writer_window:
            attention = self.layers[layer_index].self_attn
            for name in ("q_proj", "k_proj", "v_proj"):
                module = getattr(attention, name, None)
                if module is None:
                    raise ValueError(f"Layer {layer_index} has no {name}")
                self.handles.append(
                    module.register_forward_hook(self._make_projection_hook(layer_index, name))
                )
            self.handles.append(
                attention.register_forward_pre_hook(
                    self._make_attention_pre_hook(layer_index), with_kwargs=True
                )
            )

    def _make_projection_hook(self, layer_index: int, name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            if name == "q_proj":
                self.cache[layer_index][name] = output[:, -1:, :].detach()
            else:
                self.cache[layer_index][name] = output.detach()

        return hook

    def _make_attention_pre_hook(self, layer_index: int):
        def hook(_module: Any, _args: Any, kwargs: dict[str, Any]):
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError("Attention did not receive rotary position embeddings")
            cos, sin = position_embeddings
            self.cache[layer_index]["cos"] = cos.detach()
            self.cache[layer_index]["sin"] = sin.detach()
            return None

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    def clear(self) -> None:
        for values in self.cache.values():
            values.clear()

    @staticmethod
    def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        left, right = tensor.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    @torch.inference_mode()
    def collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.clear()
        decoder = getattr(self.model, "model", None)
        if decoder is None:
            raise ValueError("Scout model does not expose its decoder as model.model")
        outputs = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )
        del outputs

        config = self.model.config
        n_query_heads = int(config.num_attention_heads)
        n_key_value_heads = int(getattr(config, "num_key_value_heads", n_query_heads))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // n_query_heads))
        if n_query_heads % n_key_value_heads:
            raise RuntimeError("Query heads are not divisible by key/value heads")
        repeat_groups = n_query_heads // n_key_value_heads
        device = input_ids.device
        document_vectors = torch.zeros(
            (len(positions), int(config.hidden_size)),
            device=device,
            dtype=torch.float32,
        )
        attention_mass = torch.zeros(len(positions), device=device, dtype=torch.float32)
        position_tensors = [
            torch.as_tensor(item, device=device, dtype=torch.long) for item in positions
        ]

        for layer_index in self.writer_window:
            cached = self.cache[layer_index]
            required = {"q_proj", "k_proj", "v_proj", "cos", "sin"}
            if set(cached) != required:
                raise RuntimeError(
                    f"Incomplete projection cache at layer {layer_index}: {sorted(cached)}"
                )
            attention_module = self.layers[layer_index].self_attn
            q = cached["q_proj"].view(1, 1, n_query_heads, head_dim).transpose(1, 2)
            sequence_length = cached["k_proj"].shape[1]
            k = cached["k_proj"].view(
                1, sequence_length, n_key_value_heads, head_dim
            ).transpose(1, 2)
            v = cached["v_proj"].view(
                1, sequence_length, n_key_value_heads, head_dim
            ).transpose(1, 2)
            cos = cached["cos"].unsqueeze(1)
            sin = cached["sin"].unsqueeze(1)
            q_cos = cos[:, :, -1:, :]
            q_sin = sin[:, :, -1:, :]
            q = q * q_cos + self._rotate_half(q) * q_sin
            k = k * cos + self._rotate_half(k) * sin
            if repeat_groups > 1:
                k = k.repeat_interleave(repeat_groups, dim=1)
                v = v.repeat_interleave(repeat_groups, dim=1)
            scaling = float(getattr(attention_module, "scaling", head_dim**-0.5))
            weights = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scaling
            weights = torch.softmax(weights, dim=-1)[0, :, 0, :]
            values = v[0].transpose(0, 1)
            output_weight = attention_module.o_proj.weight
            aggregated: list[torch.Tensor] = []
            for document_positions in position_tensors:
                document_positions = document_positions[
                    document_positions < sequence_length
                ]
                if document_positions.numel() == 0:
                    aggregated.append(
                        torch.zeros(
                            n_query_heads * head_dim,
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    continue
                selected_values = values.index_select(0, document_positions)
                selected_attention = weights.index_select(1, document_positions).T
                weighted = selected_values.float() * selected_attention.unsqueeze(-1)
                aggregated.append(weighted.sum(dim=0).reshape(-1))
            head_space = torch.stack(aggregated).float()
            document_vectors += torch.matmul(head_space, output_weight.float().T)
            for document_index, document_positions in enumerate(position_tensors):
                document_positions = document_positions[
                    document_positions < sequence_length
                ]
                if document_positions.numel():
                    attention_mass[document_index] += weights.index_select(
                        1, document_positions
                    ).sum()

        token_lengths = np.asarray([len(item) for item in positions], dtype=np.int64)
        result = (
            document_vectors.cpu().numpy(),
            attention_mass.cpu().numpy(),
            token_lengths,
        )
        del document_vectors, attention_mass
        self.clear()
        return result


def extract_document_features(
    extractor: DocumentFeatureExtractor,
    loaded: LoadedModel,
    row: dict[str, Any],
    max_prompt_tokens: int,
) -> DocumentFeatures | None:
    prompt, spans = render_prompt(loaded.tokenizer, row)
    encoded = loaded.tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    prompt_tokens = int(encoded["input_ids"].shape[1])
    if prompt_tokens > max_prompt_tokens:
        return None
    offsets = [tuple(item) for item in encoded.pop("offset_mapping")[0].tolist()]
    positions_by_document = token_positions(offsets, spans)
    document_ids = [str(document["document_id"]) for document in row["documents"]]
    if any(not positions_by_document.get(document_id) for document_id in document_ids):
        raise RuntimeError(f"Empty document span in {row['example_id']}")
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
        loaded.device
    )
    vectors, attention_mass, token_lengths = extractor.collect(
        input_ids,
        attention_mask,
        [positions_by_document[document_id] for document_id in document_ids],
    )
    return DocumentFeatures(
        prompt=prompt,
        document_ids=document_ids,
        vectors=vectors,
        attention_mass=attention_mass,
        token_lengths=token_lengths,
        prompt_tokens=prompt_tokens,
    )


def extract_document_features_final_query(
    extractor: FinalQueryDocumentFeatureExtractor,
    loaded: LoadedModel,
    row: dict[str, Any],
    max_prompt_tokens: int,
) -> DocumentFeatures | None:
    """Long-context counterpart to :func:`extract_document_features`."""

    prompt, spans = render_prompt(loaded.tokenizer, row)
    encoded = loaded.tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    prompt_tokens = int(encoded["input_ids"].shape[1])
    if prompt_tokens > max_prompt_tokens:
        return None
    offsets = [tuple(item) for item in encoded.pop("offset_mapping")[0].tolist()]
    positions_by_document = token_positions(offsets, spans)
    document_ids = [str(document["document_id"]) for document in row["documents"]]
    if any(not positions_by_document.get(document_id) for document_id in document_ids):
        raise RuntimeError(f"Empty document span in {row['example_id']}")
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
        loaded.device
    )
    vectors, attention_mass, token_lengths = extractor.collect(
        input_ids,
        attention_mask,
        [positions_by_document[document_id] for document_id in document_ids],
    )
    return DocumentFeatures(
        prompt=prompt,
        document_ids=document_ids,
        vectors=vectors,
        attention_mass=attention_mass,
        token_lengths=token_lengths,
        prompt_tokens=prompt_tokens,
    )


def value_write_scores(
    features: DocumentFeatures, direction: FrozenDirection
) -> np.ndarray:
    centered = features.vectors.astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    return direction.sign * ((centered - direction.mean) @ direction.component)
