from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

import config

TOKEN_RE = re.compile(r"[A-Za-z]+(?:[.-][A-Za-z]+)*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding).replace("\r\n", "\n")
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def split_sentences(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", x).strip() for x in SENTENCE_RE.split(text)
            if len(re.sub(r"\s+", " ", x).strip()) >= 20]


def tokens(text: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(text)]


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def align_distributions(p: dict[str, float], q: dict[str, float]):
    vocabulary = sorted(set(p) | set(q))
    pv = np.array([p.get(x, 0.0) for x in vocabulary], dtype=float) + 1e-12
    qv = np.array([q.get(x, 0.0) for x in vocabulary], dtype=float) + 1e-12
    return pv / pv.sum(), qv / qv.sum()


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    pv, qv = align_distributions(p, q)
    middle = 0.5 * (pv + qv)
    return float(0.5 * np.sum(pv * np.log2(pv / middle)) +
                 0.5 * np.sum(qv * np.log2(qv / middle)))


def chunks(values: list, size: int) -> Iterable[list]:
    for start in range(0, len(values), size):
        yield values[start:start + size]
