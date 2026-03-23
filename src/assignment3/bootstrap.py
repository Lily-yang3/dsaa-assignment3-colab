from __future__ import annotations

import os
import subprocess


def _pick_gpu_from_nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    best_index: str | None = None
    best_free_memory = -1
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        index, free_memory_text = parts
        try:
            free_memory = int(free_memory_text)
        except ValueError:
            continue
        if free_memory > best_free_memory:
            best_index = index
            best_free_memory = free_memory
    return best_index


def pin_best_visible_gpu() -> str | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]

    explicit_gpu = os.environ.get("ASSIGNMENT3_GPU")
    if explicit_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = explicit_gpu
        return explicit_gpu

    best_gpu = _pick_gpu_from_nvidia_smi()
    if best_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu
    return best_gpu

