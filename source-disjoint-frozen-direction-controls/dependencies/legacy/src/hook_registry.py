from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import PIL.Image
import torch
import torch.nn as nn

if not hasattr(PIL.Image, "Resampling"):
    class _ImageResamplingCompat:
        NEAREST = PIL.Image.NEAREST
        BOX = getattr(PIL.Image, "BOX", PIL.Image.NEAREST)
        BILINEAR = PIL.Image.BILINEAR
        HAMMING = getattr(PIL.Image, "HAMMING", PIL.Image.BILINEAR)
        BICUBIC = PIL.Image.BICUBIC
        LANCZOS = getattr(PIL.Image, "LANCZOS", PIL.Image.BICUBIC)

    PIL.Image.Resampling = _ImageResamplingCompat

from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb as neox_apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from transformers.models.stablelm.modeling_stablelm import apply_rotary_pos_emb as stablelm_apply_rotary_pos_emb


def _apply_rotary_pos_emb_compat(
    apply_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle the transformers RoPE API before/after the position_ids signature change."""
    try:
        return apply_fn(q, k, cos, sin, position_ids)
    except TypeError:
        return apply_fn(q, k, cos, sin)


def _call_rotary_emb_compat(
    rotary_emb: nn.Module,
    x: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle rotary embedding forward signatures across transformers releases."""
    try:
        return rotary_emb(x, position_ids)
    except TypeError:
        try:
            return rotary_emb(x, position_ids=position_ids)
        except TypeError:
            try:
                return rotary_emb(x, seq_len=seq_len)
            except TypeError:
                return rotary_emb(x, seq_len)


def _build_rotate_half_matrix(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Builds the linear map P where rope(x) = x @ P for rotate_half-style RoPE."""
    rotary_dim = cos.shape[0]
    if rotary_dim % 2 != 0:
        raise ValueError(f"Rotary dimension must be even, got {rotary_dim}")

    half = rotary_dim // 2
    matrix = torch.zeros(rotary_dim, rotary_dim, device=cos.device, dtype=cos.dtype)
    matrix[:half, :half] = torch.diag(cos[:half])
    matrix[:half, half:] = -torch.diag(sin[:half])
    matrix[half:, :half] = torch.diag(sin[half:])
    matrix[half:, half:] = torch.diag(cos[half:])
    return matrix


def _effective_rank(values: torch.Tensor, eps: float = 1e-12) -> float:
    s = values.abs()
    total = torch.sum(s)
    if float(total) <= eps:
        return 0.0
    probs = s / total
    entropy = -(probs * torch.log(probs + eps)).sum()
    return float(torch.exp(entropy).item())


@dataclass
class LayerDecomposition:
    attn_input: torch.Tensor  # [seq, d_model]
    q: torch.Tensor  # [q_heads, seq, d_head]
    k: torch.Tensor  # [kv_heads, seq, d_head]
    v: torch.Tensor  # [kv_heads, seq, d_head]
    q_weight: torch.Tensor  # [q_heads, d_model, d_head]
    o_weight: torch.Tensor  # [q_heads, d_head, d_model]
    o_bias: Optional[torch.Tensor]  # [d_model]
    q_rotation_cache: Optional[Dict[str, torch.Tensor]]


class ArchitectureAdapter:
    name: str = "base"

    def get_base_model(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def get_layers(self, base_model: nn.Module) -> List[nn.Module]:
        raise NotImplementedError

    def get_layer_attention(self, layer: nn.Module) -> nn.Module:
        raise NotImplementedError

    def attn_input(self, layer: nn.Module, resid_pre: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decompose_layer(
        self,
        layer: nn.Module,
        attn_input: torch.Tensor,
        position_ids: torch.Tensor,
        base_model: Optional[nn.Module] = None,
    ) -> LayerDecomposition:
        raise NotImplementedError

    def num_q_heads(self, layer: nn.Module) -> int:
        raise NotImplementedError

    def num_kv_heads(self, layer: nn.Module) -> int:
        n_q = self.num_q_heads(layer)
        return n_q

    def head_dim(self, layer: nn.Module) -> int:
        raise NotImplementedError

    def q_to_kv_head(self, layer: nn.Module, q_head: int) -> int:
        n_q = self.num_q_heads(layer)
        n_kv = self.num_kv_heads(layer)
        if n_q % n_kv != 0:
            raise ValueError(f"num_q_heads ({n_q}) must be divisible by num_kv_heads ({n_kv})")
        return q_head // (n_q // n_kv)

    def query_rotation_matrix(
        self,
        layer: nn.Module,
        layer_decomp: LayerDecomposition,
        target_pos: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        d_head = self.head_dim(layer)
        device = layer_decomp.q_weight.device
        dtype = layer_decomp.q_weight.dtype
        return torch.eye(d_head, device=device, dtype=dtype)


class GPTNeoXAdapter(ArchitectureAdapter):
    name = "gpt_neox"

    @staticmethod
    def _num_attention_heads(attn: nn.Module) -> int:
        if hasattr(attn, "num_attention_heads"):
            return int(attn.num_attention_heads)
        if hasattr(attn, "config") and hasattr(attn.config, "num_attention_heads"):
            return int(attn.config.num_attention_heads)
        raise AttributeError("GPTNeoXAttention is missing num_attention_heads and config.num_attention_heads")

    def get_base_model(self, model: nn.Module) -> nn.Module:
        if not hasattr(model, "gpt_neox"):
            raise ValueError("Expected GPTNeoXForCausalLM-like model with .gpt_neox")
        return model.gpt_neox

    def get_layers(self, base_model: nn.Module) -> List[nn.Module]:
        return list(base_model.layers)

    def get_layer_attention(self, layer: nn.Module) -> nn.Module:
        return layer.attention

    def attn_input(self, layer: nn.Module, resid_pre: torch.Tensor) -> torch.Tensor:
        return layer.input_layernorm(resid_pre)

    def num_q_heads(self, layer: nn.Module) -> int:
        return self._num_attention_heads(layer.attention)

    def num_kv_heads(self, layer: nn.Module) -> int:
        return self._num_attention_heads(layer.attention)

    def head_dim(self, layer: nn.Module) -> int:
        return int(layer.attention.head_size)

    def decompose_layer(
        self,
        layer: nn.Module,
        attn_input: torch.Tensor,
        position_ids: torch.Tensor,
        base_model: Optional[nn.Module] = None,
    ) -> LayerDecomposition:
        attn = layer.attention
        bsz, seq_len, d_model = attn_input.shape
        num_heads = self._num_attention_heads(attn)
        qkv = attn.query_key_value(attn_input)
        qkv = qkv.view(bsz, seq_len, num_heads, 3 * attn.head_size).permute(0, 2, 1, 3)
        query, key, value = qkv.chunk(3, dim=-1)

        rotary_ndims = attn.rotary_ndims
        if rotary_ndims > 0:
            if hasattr(attn, "rotary_emb"):
                cos, sin = _call_rotary_emb_compat(
                    attn.rotary_emb,
                    value,
                    position_ids=position_ids,
                    seq_len=seq_len,
                )
            else:
                if base_model is None or not hasattr(base_model, "rotary_emb"):
                    raise AttributeError("GPTNeoX rotary embeddings unavailable on both attention module and base model")
                cos, sin = _call_rotary_emb_compat(
                    base_model.rotary_emb,
                    attn_input,
                    position_ids=position_ids,
                    seq_len=seq_len,
                )
            query_rot, query_pass = query[..., :rotary_ndims], query[..., rotary_ndims:]
            key_rot, key_pass = key[..., :rotary_ndims], key[..., rotary_ndims:]
            query_rot, key_rot = _apply_rotary_pos_emb_compat(
                neox_apply_rotary_pos_emb,
                query_rot,
                key_rot,
                cos,
                sin,
                position_ids,
            )
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            cos = sin = None

        qkv_weight = attn.query_key_value.weight.T.view(d_model, num_heads, 3, attn.head_size)
        q_weight = qkv_weight[:, :, 0, :].permute(1, 0, 2).contiguous()

        dense_weight = attn.dense.weight
        o_weight = []
        for h in range(num_heads):
            start = h * attn.head_size
            end = (h + 1) * attn.head_size
            block = dense_weight[:, start:end].T
            o_weight.append(block)
        o_weight = torch.stack(o_weight, dim=0)

        return LayerDecomposition(
            attn_input=attn_input[0],
            q=query[0],
            k=key[0],
            v=value[0],
            q_weight=q_weight,
            o_weight=o_weight,
            o_bias=attn.dense.bias.detach().clone() if attn.dense.bias is not None else None,
            q_rotation_cache={
                "cos": cos.detach() if cos is not None else None,
                "sin": sin.detach() if sin is not None else None,
                "rotary_ndims": torch.tensor(rotary_ndims, device=query.device),
            },
        )

    def query_rotation_matrix(
        self,
        layer: nn.Module,
        layer_decomp: LayerDecomposition,
        target_pos: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cache = layer_decomp.q_rotation_cache
        if cache is None:
            return super().query_rotation_matrix(layer, layer_decomp, target_pos, position_ids)

        d_head = self.head_dim(layer)
        rotary_ndims = int(cache["rotary_ndims"].item())
        if rotary_ndims == 0 or cache["cos"] is None or cache["sin"] is None:
            return torch.eye(d_head, device=layer_decomp.q_weight.device, dtype=layer_decomp.q_weight.dtype)
        pos = int(position_ids[0, target_pos].item())

        cos = cache["cos"]
        sin = cache["sin"]

        if cos.dim() == 4:
            cos_seq = cos[0, 0]
            sin_seq = sin[0, 0]
        elif cos.dim() == 3:
            cos_seq = cos[0]
            sin_seq = sin[0]
        elif cos.dim() == 2:
            cos_seq = cos
            sin_seq = sin
        else:
            raise ValueError(f"Unexpected RoPE cache shape: cos={tuple(cos.shape)}")

        cos_t = cos_seq[pos, :rotary_ndims]
        sin_t = sin_seq[pos, :rotary_ndims]
        rot_matrix = _build_rotate_half_matrix(cos_t, sin_t)

        full = torch.eye(d_head, device=rot_matrix.device, dtype=rot_matrix.dtype)
        full[:rotary_ndims, :rotary_ndims] = rot_matrix
        return full


class LlamaAdapter(ArchitectureAdapter):
    name = "llama"

    def get_base_model(self, model: nn.Module) -> nn.Module:
        if not hasattr(model, "model"):
            raise ValueError("Expected LlamaForCausalLM-like model with .model")
        return model.model

    def get_layers(self, base_model: nn.Module) -> List[nn.Module]:
        return list(base_model.layers)

    def get_layer_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def attn_input(self, layer: nn.Module, resid_pre: torch.Tensor) -> torch.Tensor:
        return layer.input_layernorm(resid_pre)

    def num_q_heads(self, layer: nn.Module) -> int:
        attn = layer.self_attn
        if hasattr(attn, "num_heads"):
            return int(attn.num_heads)
        if hasattr(attn, "q_proj") and hasattr(attn, "head_dim"):
            return int(attn.q_proj.out_features // attn.head_dim)
        raise AttributeError("LlamaAttention is missing both num_heads and q_proj/head_dim metadata")

    def num_kv_heads(self, layer: nn.Module) -> int:
        attn = layer.self_attn
        if hasattr(attn, "num_key_value_heads"):
            return int(attn.num_key_value_heads)
        q_heads = self.num_q_heads(layer)
        return int(attn.k_proj.out_features // attn.head_dim) if hasattr(attn, "k_proj") else q_heads

    def head_dim(self, layer: nn.Module) -> int:
        return int(layer.self_attn.head_dim)

    def decompose_layer(
        self,
        layer: nn.Module,
        attn_input: torch.Tensor,
        position_ids: torch.Tensor,
        base_model: Optional[nn.Module] = None,
    ) -> LayerDecomposition:
        attn = layer.self_attn
        bsz, seq_len, d_model = attn_input.shape
        num_q_heads = self.num_q_heads(layer)
        num_kv_heads = self.num_kv_heads(layer)
        head_dim = attn.head_dim

        q = attn.q_proj(attn_input).view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(attn_input).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(attn_input).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        rotary_emb = getattr(attn, "rotary_emb", None)
        if rotary_emb is None:
            if base_model is None or not hasattr(base_model, "rotary_emb"):
                raise AttributeError("Llama rotary embeddings unavailable on both attention module and base model")
            rotary_emb = base_model.rotary_emb
        cos, sin = _call_rotary_emb_compat(
            rotary_emb,
            v,
            position_ids=position_ids,
            seq_len=kv_seq_len,
        )
        q, k = _apply_rotary_pos_emb_compat(
            llama_apply_rotary_pos_emb,
            q,
            k,
            cos,
            sin,
            position_ids,
        )

        q_weight = attn.q_proj.weight.T.view(d_model, num_q_heads, head_dim).permute(1, 0, 2).contiguous()

        o_weight = []
        for h in range(num_q_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            block = attn.o_proj.weight[:, start:end].T
            o_weight.append(block)
        o_weight = torch.stack(o_weight, dim=0)

        return LayerDecomposition(
            attn_input=attn_input[0],
            q=q[0],
            k=k[0],
            v=v[0],
            q_weight=q_weight,
            o_weight=o_weight,
            o_bias=attn.o_proj.bias.detach().clone() if attn.o_proj.bias is not None else None,
            q_rotation_cache={"cos": cos.detach(), "sin": sin.detach()},
        )

    def query_rotation_matrix(
        self,
        layer: nn.Module,
        layer_decomp: LayerDecomposition,
        target_pos: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cache = layer_decomp.q_rotation_cache
        if cache is None:
            return super().query_rotation_matrix(layer, layer_decomp, target_pos, position_ids)

        d_head = self.head_dim(layer)
        pos = int(position_ids[0, target_pos].item())

        cos = cache["cos"]
        sin = cache["sin"]

        if cos.dim() == 4:
            cos_seq = cos[0, 0]
            sin_seq = sin[0, 0]
        elif cos.dim() == 3:
            cos_seq = cos[0]
            sin_seq = sin[0]
        elif cos.dim() == 2:
            cos_seq = cos
            sin_seq = sin
        else:
            raise ValueError(f"Unexpected RoPE cache shape: cos={tuple(cos.shape)}")

        cos_t = cos_seq[pos, :d_head]
        sin_t = sin_seq[pos, :d_head]
        return _build_rotate_half_matrix(cos_t, sin_t)


class Olmo2Adapter(LlamaAdapter):
    name = "olmo2"

    def attn_input(self, layer: nn.Module, resid_pre: torch.Tensor) -> torch.Tensor:
        # OLMo2 applies self-attention directly to the residual stream and
        # normalizes only after the attention update.
        return resid_pre


class Phi3Adapter(LlamaAdapter):
    name = "phi3"

    def num_q_heads(self, layer: nn.Module) -> int:
        attn = layer.self_attn
        if hasattr(attn, "num_heads"):
            return int(attn.num_heads)
        raise AttributeError("Phi3Attention is missing num_heads metadata")

    def num_kv_heads(self, layer: nn.Module) -> int:
        attn = layer.self_attn
        if hasattr(attn, "num_key_value_heads"):
            return int(attn.num_key_value_heads)
        return self.num_q_heads(layer)

    def decompose_layer(
        self,
        layer: nn.Module,
        attn_input: torch.Tensor,
        position_ids: torch.Tensor,
        base_model: Optional[nn.Module] = None,
    ) -> LayerDecomposition:
        del base_model
        attn = layer.self_attn
        bsz, seq_len, d_model = attn_input.shape
        num_q_heads = self.num_q_heads(layer)
        num_kv_heads = self.num_kv_heads(layer)
        head_dim = attn.head_dim

        qkv = attn.qkv_proj(attn_input)
        query_pos = num_q_heads * head_dim
        kv_width = num_kv_heads * head_dim
        q = qkv[..., :query_pos].view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = qkv[..., query_pos : query_pos + kv_width].view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = qkv[..., query_pos + kv_width :].view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        cos, sin = _call_rotary_emb_compat(
            attn.rotary_emb,
            v,
            position_ids=position_ids,
            seq_len=kv_seq_len,
        )
        q, k = _apply_rotary_pos_emb_compat(
            llama_apply_rotary_pos_emb,
            q,
            k,
            cos,
            sin,
            position_ids,
        )

        q_weight = (
            attn.qkv_proj.weight[:query_pos, :]
            .T.view(d_model, num_q_heads, head_dim)
            .permute(1, 0, 2)
            .contiguous()
        )

        o_weight = []
        for h in range(num_q_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            block = attn.o_proj.weight[:, start:end].T
            o_weight.append(block)
        o_weight = torch.stack(o_weight, dim=0)

        return LayerDecomposition(
            attn_input=attn_input[0],
            q=q[0],
            k=k[0],
            v=v[0],
            q_weight=q_weight,
            o_weight=o_weight,
            o_bias=attn.o_proj.bias.detach().clone() if attn.o_proj.bias is not None else None,
            q_rotation_cache={"cos": cos.detach(), "sin": sin.detach()},
        )


class StableLMAdapter(LlamaAdapter):
    name = "stablelm"

    def decompose_layer(
        self,
        layer: nn.Module,
        attn_input: torch.Tensor,
        position_ids: torch.Tensor,
        base_model: Optional[nn.Module] = None,
    ) -> LayerDecomposition:
        attn = layer.self_attn
        bsz, seq_len, d_model = attn_input.shape
        num_q_heads = self.num_q_heads(layer)
        num_kv_heads = self.num_kv_heads(layer)
        head_dim = attn.head_dim

        q = attn.q_proj(attn_input).view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(attn_input).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(attn_input).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        if getattr(attn, "qk_layernorm", False):
            q = attn.q_layernorm(q)
            k = attn.k_layernorm(k)

        rotary_ndims = int(getattr(attn, "rotary_ndims", 0))
        if rotary_ndims > 0:
            rotary_emb = getattr(attn, "rotary_emb", None)
            if rotary_emb is None:
                if base_model is None or not hasattr(base_model, "rotary_emb"):
                    raise AttributeError("StableLM rotary embeddings unavailable on both attention module and base model")
                rotary_emb = base_model.rotary_emb
            cos, sin = _call_rotary_emb_compat(
                rotary_emb,
                attn_input,
                position_ids=position_ids,
                seq_len=seq_len,
            )
            q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
            k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
            q_rot, k_rot = stablelm_apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        else:
            cos = sin = None

        q_weight = attn.q_proj.weight.T.view(d_model, num_q_heads, head_dim).permute(1, 0, 2).contiguous()

        o_weight = []
        for h in range(num_q_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            block = attn.o_proj.weight[:, start:end].T
            o_weight.append(block)
        o_weight = torch.stack(o_weight, dim=0)

        return LayerDecomposition(
            attn_input=attn_input[0],
            q=q[0],
            k=k[0],
            v=v[0],
            q_weight=q_weight,
            o_weight=o_weight,
            o_bias=attn.o_proj.bias.detach().clone() if attn.o_proj.bias is not None else None,
            q_rotation_cache={
                "cos": cos.detach() if cos is not None else None,
                "sin": sin.detach() if sin is not None else None,
                "rotary_ndims": torch.tensor(rotary_ndims, device=q.device),
            },
        )

    def query_rotation_matrix(
        self,
        layer: nn.Module,
        layer_decomp: LayerDecomposition,
        target_pos: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cache = layer_decomp.q_rotation_cache
        if cache is None:
            return super().query_rotation_matrix(layer, layer_decomp, target_pos, position_ids)

        d_head = self.head_dim(layer)
        rotary_ndims = int(cache["rotary_ndims"].item())
        if rotary_ndims == 0 or cache["cos"] is None or cache["sin"] is None:
            return torch.eye(d_head, device=layer_decomp.q_weight.device, dtype=layer_decomp.q_weight.dtype)
        pos = int(position_ids[0, target_pos].item())

        cos = cache["cos"]
        sin = cache["sin"]

        if cos.dim() == 4:
            cos_seq = cos[0, 0]
            sin_seq = sin[0, 0]
        elif cos.dim() == 3:
            cos_seq = cos[0]
            sin_seq = sin[0]
        elif cos.dim() == 2:
            cos_seq = cos
            sin_seq = sin
        else:
            raise ValueError(f"Unexpected RoPE cache shape: cos={tuple(cos.shape)}")

        cos_t = cos_seq[pos, :rotary_ndims]
        sin_t = sin_seq[pos, :rotary_ndims]
        rot_matrix = _build_rotate_half_matrix(cos_t, sin_t)

        full = torch.eye(d_head, device=rot_matrix.device, dtype=rot_matrix.dtype)
        full[:rotary_ndims, :rotary_ndims] = rot_matrix
        return full


class StableLMAlphaAdapter(ArchitectureAdapter):
    name = "stablelm_alpha"

    def get_base_model(self, model: nn.Module) -> nn.Module:
        if not hasattr(model, "transformer"):
            raise ValueError("Expected StableLMAlphaForCausalLM-like model with .transformer")
        return model.transformer

    def get_layers(self, base_model: nn.Module) -> List[nn.Module]:
        return list(base_model.layers)

    def get_layer_attention(self, layer: nn.Module) -> nn.Module:
        return layer.attention

    def attn_input(self, layer: nn.Module, resid_pre: torch.Tensor) -> torch.Tensor:
        return layer.norm(resid_pre)

    def num_q_heads(self, layer: nn.Module) -> int:
        return int(layer.attention.num_heads)

    def num_kv_heads(self, layer: nn.Module) -> int:
        return int(layer.attention.num_heads)

    def head_dim(self, layer: nn.Module) -> int:
        return int(layer.attention.head_dim)

    def decompose_layer(
        self,
        layer: nn.Module,
        attn_input: torch.Tensor,
        position_ids: torch.Tensor,
        base_model: Optional[nn.Module] = None,
    ) -> LayerDecomposition:
        del base_model
        attn = layer.attention
        bsz, seq_len, d_model = attn_input.shape
        num_heads = self.num_q_heads(layer)
        head_dim = self.head_dim(layer)

        qkv = attn.qkv_proj(attn_input)
        qkv = qkv.view(bsz, seq_len, num_heads, 3 * head_dim)
        query = qkv[..., :head_dim].permute(0, 2, 1, 3)
        key = qkv[..., head_dim : 2 * head_dim].permute(0, 2, 1, 3)
        value = qkv[..., 2 * head_dim :].permute(0, 2, 1, 3)

        rotary_ndims = int(attn.rotary_ndims)
        if rotary_ndims > 0:
            query_rot, query_pass = query[..., :rotary_ndims], query[..., rotary_ndims:]
            key_rot, key_pass = key[..., :rotary_ndims], key[..., rotary_ndims:]
            cos, sin = _call_rotary_emb_compat(
                attn.rotary_emb,
                value,
                position_ids=position_ids,
                seq_len=seq_len,
            )
            query_rot, key_rot = _apply_rotary_pos_emb_compat(
                llama_apply_rotary_pos_emb,
                query_rot,
                key_rot,
                cos,
                sin,
                position_ids,
            )
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            cos = sin = None

        qkv_weight = attn.qkv_proj.weight.T.view(d_model, num_heads, 3, head_dim)
        q_weight = qkv_weight[:, :, 0, :].permute(1, 0, 2).contiguous()

        out_proj_weight = attn.out_proj.weight
        o_weight = []
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            block = out_proj_weight[:, start:end].T
            o_weight.append(block)
        o_weight = torch.stack(o_weight, dim=0)

        return LayerDecomposition(
            attn_input=attn_input[0],
            q=query[0],
            k=key[0],
            v=value[0],
            q_weight=q_weight,
            o_weight=o_weight,
            o_bias=attn.out_proj.bias.detach().clone() if attn.out_proj.bias is not None else None,
            q_rotation_cache={"cos": cos.detach(), "sin": sin.detach()} if cos is not None and sin is not None else None,
        )


class GPT2Adapter(ArchitectureAdapter):
    name = "gpt2"

    def get_base_model(self, model: nn.Module) -> nn.Module:
        if not hasattr(model, "transformer"):
            raise ValueError("Expected GPT2LMHeadModel-like model with .transformer")
        return model.transformer

    def get_layers(self, base_model: nn.Module) -> List[nn.Module]:
        return list(base_model.h)

    def get_layer_attention(self, layer: nn.Module) -> nn.Module:
        return layer.attn

    def attn_input(self, layer: nn.Module, resid_pre: torch.Tensor) -> torch.Tensor:
        return layer.ln_1(resid_pre)

    def num_q_heads(self, layer: nn.Module) -> int:
        return int(layer.attn.num_heads)

    def head_dim(self, layer: nn.Module) -> int:
        return int(layer.attn.head_dim)

    def decompose_layer(
        self,
        layer: nn.Module,
        attn_input: torch.Tensor,
        position_ids: torch.Tensor,
        base_model: Optional[nn.Module] = None,
    ) -> LayerDecomposition:
        del base_model
        del position_ids
        attn = layer.attn
        bsz, seq_len, d_model = attn_input.shape
        num_heads = attn.num_heads
        head_dim = attn.head_dim

        qkv = attn.c_attn(attn_input)
        query, key, value = qkv.split(attn.split_size, dim=2)
        query = query.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        key = key.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        q_weight_full = attn.c_attn.weight[:, :d_model]
        q_weight = q_weight_full.view(d_model, num_heads, head_dim).permute(1, 0, 2).contiguous()

        o_weight = []
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            block = attn.c_proj.weight[start:end, :]
            o_weight.append(block)
        o_weight = torch.stack(o_weight, dim=0)

        return LayerDecomposition(
            attn_input=attn_input[0],
            q=query[0],
            k=key[0],
            v=value[0],
            q_weight=q_weight,
            o_weight=o_weight,
            o_bias=attn.c_proj.bias.detach().clone() if attn.c_proj.bias is not None else None,
            q_rotation_cache=None,
        )


def resolve_adapter(model: nn.Module) -> ArchitectureAdapter:
    model_type = getattr(model.config, "model_type", "")

    if model_type == "gpt_neox":
        return GPTNeoXAdapter()
    if model_type == "llama":
        return LlamaAdapter()
    if model_type == "mistral":
        return LlamaAdapter()
    if model_type == "ministral":
        return LlamaAdapter()
    if model_type == "qwen2":
        return LlamaAdapter()
    if model_type == "stablelm":
        return StableLMAdapter()
    if model_type == "granite":
        return LlamaAdapter()
    if model_type == "olmo2":
        return Olmo2Adapter()
    if model_type == "stablelm_alpha":
        return StableLMAlphaAdapter()
    if model_type == "phi3":
        return Phi3Adapter()
    if model_type == "phi3small":
        return Phi3Adapter()
    if model_type == "gpt2":
        return GPT2Adapter()

    # Conservative fallback on architecture name.
    archs = getattr(model.config, "architectures", []) or []
    arch_text = " ".join(archs).lower()
    if "neox" in arch_text:
        return GPTNeoXAdapter()
    if "llama" in arch_text:
        return LlamaAdapter()
    if "qwen2" in arch_text or "qwen" in arch_text:
        return LlamaAdapter()
    if "stablelm" in arch_text or "stable lm" in arch_text:
        return LlamaAdapter()
    if "granite" in arch_text:
        return LlamaAdapter()
    if "olmo2" in arch_text or "olmo" in arch_text:
        return Olmo2Adapter()
    if "stablelmalpha" in arch_text or "stablelm alpha" in arch_text:
        return StableLMAlphaAdapter()
    if "phi3small" in arch_text or "phi3" in arch_text or "phi-3" in arch_text:
        return Phi3Adapter()
    if "gpt2" in arch_text:
        return GPT2Adapter()

    raise ValueError(
        "Unsupported architecture for dynamic MLP decomposition. "
        f"model_type={model_type}, architectures={archs}"
    )


def compute_operator_metrics(w1: torch.Tensor, w2: torch.Tensor, top_k: int = 8) -> Dict[str, object]:
    s1 = torch.linalg.svdvals(w1)
    s2 = torch.linalg.svdvals(w2)

    top1 = s1[:top_k].detach().cpu().tolist()
    top2 = s2[:top_k].detach().cpu().tolist()

    eps = 1e-12
    cond1 = float((s1[0] / torch.clamp(s1[-1], min=eps)).item()) if s1.numel() > 0 else float("nan")
    cond2 = float((s2[0] / torch.clamp(s2[-1], min=eps)).item()) if s2.numel() > 0 else float("nan")

    return {
        "eff_rank_w1": _effective_rank(s1),
        "eff_rank_w2": _effective_rank(s2),
        "eff_rank_j": float("nan"),
        "top_sv_w1_json": top1,
        "top_sv_w2_json": top2,
        "top_sv_j_json": [],
        "cond_w1": cond1,
        "cond_w2": cond2,
        "cond_j": float("nan"),
    }
