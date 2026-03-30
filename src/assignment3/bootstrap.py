from __future__ import annotations

import os
import subprocess


def _pick_gpu_from_nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    candidates: list[tuple[int, int, str]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, free_memory_text, utilization_text = parts
        try:
            free_memory = int(free_memory_text)
            utilization = int(utilization_text)
        except ValueError:
            continue
        candidates.append((utilization, -free_memory, index))

    if not candidates:
        return None

    low_utilization = [candidate for candidate in candidates if candidate[0] <= 10]
    pool = low_utilization if low_utilization else candidates
    pool.sort()
    return pool[0][2]


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
