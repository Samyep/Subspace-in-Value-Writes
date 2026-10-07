from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_mean(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    remaining = draws
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, len(array), size=(batch, len(array)))
        samples.append(array[indices].mean(axis=1))
        remaining -= batch
    distribution = np.concatenate(samples)
    return {
        "mean": float(array.mean()),
        "lo": float(np.percentile(distribution, 2.5)),
        "hi": float(np.percentile(distribution, 97.5)),
        "n": int(len(array)),
    }


def summarize_metrics(
    rows: list[dict[str, Any]], keys: list[str], draws: int, seed: int
) -> dict[str, Any]:
    return {
        key: bootstrap_mean(
            [float(row[key]) for row in rows if row.get(key) is not None],
            draws,
            seed + index,
        )
        for index, key in enumerate(keys)
    }
