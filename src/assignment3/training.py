from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from assignment3.models import count_trainable_parameters, load_tokenizer, load_training_model
from assignment3.runtime import ensure_parent_dir, get_device_info, seed_everything, write_json


@dataclass(slots=True)
class SupervisedExample:
    prompt: str
    target: str


class SupervisedPairDataset(Dataset):
    def __init__(
        self,
        examples: list[SupervisedExample],
        tokenizer: Any,
        max_seq_length: int,
        max_target_tokens: int,
    ) -> None:
        self.features = [
            tokenize_supervised_example(
                prompt=example.prompt,
                target=example.target,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_target_tokens=max_target_tokens,
            )
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_supervised_example(
    prompt: str,
    target: str,
    tokenizer: Any,
    max_seq_length: int,
    max_target_tokens: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids[: max(1, max_target_tokens - 1)] + [tokenizer.eos_token_id]
    else:
        target_ids = target_ids[:max_target_tokens]

    available_prompt_tokens = max_seq_length - len(target_ids)
    if available_prompt_tokens <= 0:
        target_ids = target_ids[: max_seq_length - 1]
        available_prompt_tokens = 1

    prompt_ids = prompt_ids[:available_prompt_tokens]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids[:max_seq_length],
        "attention_mask": attention_mask[:max_seq_length],
        "labels": labels[:max_seq_length],
    }


def build_trainer(
    model_name: str,
    runtime_config: dict[str, Any],
    lora_config: dict[str, Any],
    training_config: dict[str, Any],
    train_examples: list[SupervisedExample],
    val_examples: list[SupervisedExample],
    max_seq_length: int,
    max_target_tokens: int,
    output_dir: str,
) -> tuple[Any, Any, Trainer]:
    tokenizer = load_tokenizer(
        model_name=model_name,
        trust_remote_code=runtime_config.get("trust_remote_code", True),
    )
    model = load_training_model(
        model_name=model_name,
        runtime_config=runtime_config,
        lora_config=lora_config,
    )
    param_counts = count_trainable_parameters(model)
    print(f"Trainable params: {param_counts['trainable']:,} / {param_counts['total']:,}")

    train_dataset = SupervisedPairDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_target_tokens=max_target_tokens,
    )
    eval_dataset = None
    if val_examples:
        eval_dataset = SupervisedPairDataset(
            examples=val_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_target_tokens=max_target_tokens,
        )

    device_info = get_device_info(runtime_config.get("preferred_device"))
    optim = "paged_adamw_8bit" if runtime_config.get("load_in_4bit", True) and device_info.cuda_available else "adamw_torch"
    training_args_kwargs = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "learning_rate": training_config["learning_rate"],
        "weight_decay": training_config.get("weight_decay", 0.0),
        "warmup_ratio": training_config.get("warmup_ratio", 0.0),
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "cosine"),
        "logging_steps": training_config.get("logging_steps", 10),
        "save_steps": training_config.get("save_steps", 100),
        "eval_steps": training_config.get("eval_steps", 100),
        "save_total_limit": training_config.get("save_total_limit", 2),
        "bf16": device_info.bf16_supported,
        "fp16": device_info.cuda_available and not device_info.bf16_supported,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": runtime_config.get("gradient_checkpointing", True),
        "dataloader_pin_memory": device_info.cuda_available,
        "optim": optim,
    }
    strategy_value = "steps" if eval_dataset is not None else "no"
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = strategy_value
    else:
        training_args_kwargs["eval_strategy"] = strategy_value

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    return model, tokenizer, trainer


def train_and_save(
    model_name: str,
    runtime_config: dict[str, Any],
    lora_config: dict[str, Any],
    training_config: dict[str, Any],
    train_examples: list[SupervisedExample],
    val_examples: list[SupervisedExample],
    max_seq_length: int,
    max_target_tokens: int,
    output_dir: str,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    seed_everything(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer, trainer = build_trainer(
        model_name=model_name,
        runtime_config=runtime_config,
        lora_config=lora_config,
        training_config=training_config,
        train_examples=train_examples,
        val_examples=val_examples,
        max_seq_length=max_seq_length,
        max_target_tokens=max_target_tokens,
        output_dir=output_dir,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = dict(train_result.metrics)
    metrics["train_examples"] = len(train_examples)
    metrics["val_examples"] = len(val_examples)

    metrics_path = ensure_parent_dir(Path(output_dir) / "train_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return model, tokenizer, metrics


def dump_examples(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_parent_dir(path)
    Path(path).write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n")


def save_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
