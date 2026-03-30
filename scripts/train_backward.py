from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment3.bootstrap import pin_best_visible_gpu

pin_best_visible_gpu()

from assignment3.data import load_json, load_seed_dataset, train_val_split
from assignment3.models import push_model_artifacts
from assignment3.prompts import format_backward_prompt
from assignment3.runtime import resolve_local_reference, runtime_summary
from assignment3.training import SupervisedExample, save_metadata, train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the backward LoRA model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/backward_train.json",
        help="Path to the JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / args.config)

    seed_rows = load_seed_dataset(
        dataset_name=config["dataset_name"],
        split=config["dataset_split"],
        seed=config["seed"],
        limit=config.get("train_subset"),
    )
    train_rows, val_rows = train_val_split(
        seed_rows,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    train_examples = [
        SupervisedExample(
            prompt=format_backward_prompt(row["response"]),
            target=row["instruction"],
        )
        for row in train_rows
    ]
    val_examples = [
        SupervisedExample(
            prompt=format_backward_prompt(row["response"]),
            target=row["instruction"],
        )
        for row in val_rows
    ]

    print(f"Loaded {len(seed_rows)} seed examples")
    print(f"Train split: {len(train_examples)}")
    print(f"Validation split: {len(val_examples)}")

    model_name = resolve_local_reference(ROOT, config["model_name"])

    model, tokenizer, metrics = train_and_save(
        model_name=model_name,
        runtime_config=config["runtime"],
        lora_config=config["lora"],
        training_config=config["training"],
        train_examples=train_examples,
        val_examples=val_examples,
        max_seq_length=config["max_seq_length"],
        max_target_tokens=config["max_target_tokens"],
        output_dir=str(ROOT / config["output_dir"]),
        seed=config["seed"],
        resume_from_checkpoint=resolve_local_reference(ROOT, config.get("resume_from_checkpoint")),
    )

    save_metadata(
        ROOT / config["output_dir"] / "runtime_summary.json",
        runtime_summary(config["runtime"].get("preferred_device")),
    )
    print("Training metrics:")
    print(metrics)

    if config.get("push_to_hub") and config.get("hub_model_id"):
        push_model_artifacts(
            model=model,
            tokenizer=tokenizer,
            repo_id=config["hub_model_id"],
            private=config.get("hub_private", False),
        )
        print(f"Pushed backward model to {config['hub_model_id']}")


if __name__ == "__main__":
    main()
