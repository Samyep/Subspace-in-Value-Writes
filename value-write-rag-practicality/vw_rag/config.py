from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    cell: str
    model_id: str
    writer_window: tuple[int, ...]
    hidden_size: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["writer_window"] = list(self.writer_window)
        return result


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "llama_3b": ModelConfig(
        cell="llama_3b",
        model_id="unsloth/Llama-3.2-3B-Instruct",
        writer_window=tuple(range(14, 18)),
        hidden_size=3072,
    ),
    "qwen_7b": ModelConfig(
        cell="qwen_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        writer_window=tuple(range(24, 28)),
        hidden_size=3584,
    ),
    "qwen_14b": ModelConfig(
        cell="qwen_14b",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        writer_window=tuple(range(32, 48)),
        hidden_size=5120,
    ),
    "qwen3_8b": ModelConfig(
        cell="qwen3_8b",
        model_id="Qwen/Qwen3-8B",
        writer_window=tuple(range(24, 36)),
        hidden_size=4096,
    ),
}


def get_model_config(cell: str, model_id_override: str | None = None) -> ModelConfig:
    try:
        config = MODEL_CONFIGS[cell]
    except KeyError as error:
        raise ValueError(f"Unknown model cell: {cell}") from error
    if model_id_override is None:
        return config
    return ModelConfig(
        cell=config.cell,
        model_id=model_id_override,
        writer_window=config.writer_window,
        hidden_size=config.hidden_size,
    )
