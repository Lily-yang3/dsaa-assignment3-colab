from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class DeviceInfo:
    device: str
    device_index: int | None
    device_name: str
    bf16_supported: bool
    cuda_available: bool
    num_gpus: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_preferred_device(preferred_device: str | None) -> int | None:
    if preferred_device is None:
        return None
    if preferred_device.isdigit():
        return int(preferred_device)
    if preferred_device.startswith("cuda:"):
        return int(preferred_device.split(":", maxsplit=1)[1])
    return None


def pick_best_cuda_device(preferred_device: str | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None

    preferred_index = _parse_preferred_device(preferred_device)
    if preferred_index is not None:
        return preferred_index

    best_index = 0
    best_free_memory = -1
    for index in range(torch.cuda.device_count()):
        try:
            free_memory, _ = torch.cuda.mem_get_info(index)
        except Exception:
            free_memory = 0
        if free_memory > best_free_memory:
            best_index = index
            best_free_memory = free_memory
    return best_index


def get_device_info(preferred_device: str | None = None) -> DeviceInfo:
    if not torch.cuda.is_available():
        return DeviceInfo(
            device="cpu",
            device_index=None,
            device_name="cpu",
            bf16_supported=False,
            cuda_available=False,
            num_gpus=0,
        )

    index = pick_best_cuda_device(preferred_device)
    assert index is not None
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return DeviceInfo(
        device=f"cuda:{index}",
        device_index=index,
        device_name=torch.cuda.get_device_name(index),
        bf16_supported=bf16_supported,
        cuda_available=True,
        num_gpus=torch.cuda.device_count(),
    )


def get_default_torch_dtype(device_info: DeviceInfo) -> torch.dtype:
    if not device_info.cuda_available:
        return torch.float32
    if device_info.bf16_supported:
        return torch.bfloat16
    return torch.float16


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def resolve_local_reference(root: str | Path, value: str | Path | None) -> str | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)

    rooted = Path(root) / path
    if rooted.exists():
        return str(rooted.resolve())

    if path.exists():
        return str(path.resolve())

    return str(value)


def runtime_summary(preferred_device: str | None = None) -> dict[str, Any]:
    info = get_device_info(preferred_device)
    dtype = str(get_default_torch_dtype(info)).replace("torch.", "")
    payload = asdict(info)
    payload["default_dtype"] = dtype
    return payload
