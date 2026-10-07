from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import torch


T = TypeVar("T")


@dataclass
class TimedResult(Generic[T]):
    value: T
    seconds: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    baseline_allocated_bytes: int | None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function: Callable[[], T]) -> TimedResult[T]:
    if device.type == "cuda":
        synchronize(device)
        baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    started = time.perf_counter()
    value = function()
    synchronize(device)
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = None
        peak_reserved = None
    return TimedResult(
        value=value,
        seconds=elapsed,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        baseline_allocated_bytes=baseline,
    )
