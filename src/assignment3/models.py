from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from assignment3.runtime import get_default_torch_dtype, get_device_info


# Some server environments stall on Hugging Face's xet-backed large-file transport.
# Falling back to the standard download path is slower in the best case but more reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def _require_peft() -> Any:
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `peft`. Install project dependencies with "
            "`pip install -e .` before training."
        ) from exc
    return LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def build_quantization_config(runtime_config: dict[str, Any]) -> BitsAndBytesConfig | None:
    device_info = get_device_info(runtime_config.get("preferred_device"))
    if not runtime_config.get("load_in_4bit", True) or not device_info.cuda_available:
        return None
    compute_dtype = get_default_torch_dtype(device_info)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _build_model_kwargs(runtime_config: dict[str, Any]) -> dict[str, Any]:
    device_info = get_device_info(runtime_config.get("preferred_device"))
    torch_dtype = get_default_torch_dtype(device_info)
    quantization_config = build_quantization_config(runtime_config)

    kwargs: dict[str, Any] = {
        "trust_remote_code": runtime_config.get("trust_remote_code", True),
    }
    if runtime_config.get("attn_implementation"):
        kwargs["attn_implementation"] = runtime_config["attn_implementation"]

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = {"": device_info.device_index}
        kwargs["torch_dtype"] = torch_dtype
    elif device_info.cuda_available:
        kwargs["torch_dtype"] = torch_dtype
    else:
        kwargs["torch_dtype"] = torch.float32

    return kwargs


def load_training_model(
    model_name: str,
    runtime_config: dict[str, Any],
    lora_config: dict[str, Any],
) -> Any:
    LoraConfig, _, get_peft_model, prepare_model_for_kbit_training = _require_peft()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_build_model_kwargs(runtime_config),
    )

    if runtime_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if runtime_config.get("load_in_4bit", True) and get_device_info(
        runtime_config.get("preferred_device")
    ).cuda_available:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config.get("bias", "none"),
        target_modules=lora_config["target_modules"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def load_inference_model(
    model_name: str,
    runtime_config: dict[str, Any],
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(
        model_name=model_name,
        trust_remote_code=runtime_config.get("trust_remote_code", True),
    )
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_build_model_kwargs(runtime_config),
    )
    if adapter_path:
        _, PeftModel, _, _ = _require_peft()
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def count_trainable_parameters(model: Any) -> dict[str, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return {"trainable": trainable, "total": total}


def push_model_artifacts(model: Any, tokenizer: Any, repo_id: str, private: bool) -> None:
    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)
