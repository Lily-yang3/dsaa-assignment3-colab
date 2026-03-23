from __future__ import annotations

from typing import Any

import torch


def _get_model_device(model: Any) -> str | torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for value in hf_device_map.values():
            if isinstance(value, int):
                return f"cuda:{value}"
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return value
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


@torch.no_grad()
def generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    generation_config: dict[str, Any],
    max_input_length: int,
) -> list[str]:
    if not prompts:
        return []

    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    device = _get_model_device(model)
    batch = {key: value.to(device) for key, value in batch.items()}

    generation_kwargs = {
        "max_new_tokens": generation_config.get("max_new_tokens", 128),
        "do_sample": generation_config.get("do_sample", True),
        "repetition_penalty": generation_config.get("repetition_penalty", 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = generation_config.get("temperature", 0.7)
        generation_kwargs["top_p"] = generation_config.get("top_p", 0.9)

    outputs = model.generate(
        **batch,
        **generation_kwargs,
    )

    prompt_length = batch["input_ids"].shape[1]
    results: list[str] = []
    for output in outputs:
        generated_ids = output[prompt_length:]
        results.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return results


def batched_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    generation_config: dict[str, Any],
    max_input_length: int,
) -> list[str]:
    batch_size = generation_config.get("batch_size", 1)
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        outputs.extend(
            generate_texts(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                generation_config=generation_config,
                max_input_length=max_input_length,
            )
        )
    return outputs
