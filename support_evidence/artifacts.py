from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def load_json(path: str | Path) -> Any:
    p = repo_path(path) if isinstance(path, str) else path
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = repo_path(path) if isinstance(path, str) else path
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: str | Path) -> list[dict[str, str]]:
    p = repo_path(path) if isinstance(path, str) else path
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def artifact_manifest() -> list[dict[str, Any]]:
    return load_json("artifacts/ARTIFACT_MANIFEST.json")
