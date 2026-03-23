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

from assignment3.data import load_json, load_single_turn_lima, write_jsonl
from assignment3.inference import batched_generate
from assignment3.models import load_inference_model
from assignment3.prompts import format_backward_prompt
from assignment3.runtime import resolve_local_reference, runtime_summary, write_json
from assignment3.training import dump_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic instructions from LIMA responses.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/augmentation.json",
        help="Path to the JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / args.config)

    rows = load_single_turn_lima(
        dataset_name=config["dataset_name"],
        split=config["dataset_split"],
        sample_size=config["sample_size"],
        seed=config["seed"],
    )
    print(f"Loaded {len(rows)} single-turn LIMA rows")

    model, tokenizer = load_inference_model(
        model_name=resolve_local_reference(ROOT, config["model_name"]),
        runtime_config=config["runtime"],
        adapter_path=resolve_local_reference(ROOT, config["adapter_path"]),
    )

    prompts = [format_backward_prompt(row["response"]) for row in rows]
    generated_instructions = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=config["generation"],
        max_input_length=config["max_input_length"],
    )

    output_rows = []
    for index, (row, generated_instruction) in enumerate(zip(rows, generated_instructions)):
        output_rows.append(
            {
                "id": index,
                "instruction": generated_instruction.strip(),
                "response": row["response"],
                "source_instruction": row["instruction"],
                "source_dataset": config["dataset_name"],
            }
        )

    output_path = ROOT / config["output_path"]
    write_jsonl(output_path, output_rows)
    write_json(
        output_path.parent / "runtime_summary.json",
        runtime_summary(config["runtime"].get("preferred_device")),
    )
    dump_examples(
        output_path.parent / "generated_examples.json",
        output_rows[: config["show_examples"]],
    )

    print(f"Saved generated pairs to {output_path}")
    print(f"Printing {config['show_examples']} generated examples")
    for example in output_rows[: config["show_examples"]]:
        print("=" * 80)
        print(f"Generated instruction: {example['instruction']}")
        print(f"Response: {example['response'][:500]}")


if __name__ == "__main__":
    main()
