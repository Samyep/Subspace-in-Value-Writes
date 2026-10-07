from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def stable_int(seed: int, namespace: str, value: str) -> int:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def portable_path(path: Path) -> str:
    """Return a repository-relative identifier without recording local paths."""
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def runtime_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as error:  # pragma: no cover - diagnostic only
        result["torch_error"] = repr(error)
    try:
        import transformers

        result["transformers"] = transformers.__version__
    except Exception as error:  # pragma: no cover - diagnostic only
        result["transformers_error"] = repr(error)
    return result


def select_shard(
    rows: list[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    ordered = sorted(rows, key=lambda row: int(row["split_rank"]))
    return [row for offset, row in enumerate(ordered) if offset % num_shards == shard_index]
