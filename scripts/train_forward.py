from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment3.bootstrap import pin_best_visible_gpu

pin_best_visible_gpu()

from assignment3.data import load_json, load_seed_dataset, read_jsonl, train_val_split
from assignment3.inference import batched_generate
from assignment3.models import load_inference_model, push_model_artifacts
from assignment3.prompts import format_forward_prompt
from assignment3.runtime import resolve_local_reference, runtime_summary
from assignment3.training import (
    SupervisedExample,
    dump_examples,
    save_metadata,
    train_and_save,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final instruction-following LoRA model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/forward_train.json",
        help="Path to the JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / args.config)

    rows = read_jsonl(ROOT / config["dataset_path"])
    rows = [{"instruction": row["instruction"], "response": row["response"]} for row in rows]

    if config.get("include_seed_dataset", False):
        rows.extend(
            load_seed_dataset(
                dataset_name=config["seed_dataset_name"],
                split=config["seed_dataset_split"],
                seed=config["seed"],
                limit=None,
            )
        )

    train_rows, val_rows = train_val_split(
        rows,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    train_examples = [
        SupervisedExample(
            prompt=format_forward_prompt(row["instruction"]),
            target=row["response"],
        )
        for row in train_rows
    ]
    val_examples = [
        SupervisedExample(
            prompt=format_forward_prompt(row["instruction"]),
            target=row["response"],
        )
        for row in val_rows
    ]

    print(f"Loaded {len(rows)} final training rows")
    print(f"Train split: {len(train_examples)}")
    print(f"Validation split: {len(val_examples)}")
    effective_batch_size = (
        config["training"]["per_device_train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
    )
    estimated_steps_per_epoch = max(1, math.ceil(len(train_examples) / effective_batch_size))
    print(f"Estimated optimizer steps per epoch: {estimated_steps_per_epoch}")

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
        print(f"Pushed final model to {config['hub_model_id']}")

    demo_source_rows = (val_rows + train_rows)[: config["show_examples"]]
    demo_prompts = [format_forward_prompt(row["instruction"]) for row in demo_source_rows]
    if demo_prompts:
        demo_model, demo_tokenizer = load_inference_model(
            model_name=model_name,
            runtime_config=config["runtime"],
            adapter_path=resolve_local_reference(ROOT, config["output_dir"]),
        )
        generations = batched_generate(
            model=demo_model,
            tokenizer=demo_tokenizer,
            prompts=demo_prompts,
            generation_config=config["demo_generation"],
            max_input_length=config["max_seq_length"],
        )
        demo_rows = []
        for row, generation in zip(demo_source_rows, generations):
            demo_rows.append(
                {
                    "instruction": row["instruction"],
                    "reference_response": row["response"],
                    "generated_response": generation,
                }
            )

        dump_examples(ROOT / config["output_dir"] / "demo_generations.json", demo_rows)
        print(f"Printing {len(demo_rows)} sample responses")
        for example in demo_rows:
            print("=" * 80)
            print(f"Instruction: {example['instruction']}")
            print(f"Generated response: {example['generated_response']}")


if __name__ == "__main__":
    main()
