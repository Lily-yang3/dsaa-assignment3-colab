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

from assignment3.data import load_json, read_jsonl, to_hf_dataset, write_jsonl
from assignment3.inference import batched_generate
from assignment3.models import load_inference_model
from assignment3.prompts import extract_reason, extract_score, format_quality_prompt
from assignment3.runtime import resolve_local_reference, runtime_summary, write_json
from assignment3.training import dump_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate generated instruction-response pairs.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curation.json",
        help="Path to the JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / args.config)
    candidate_rows = read_jsonl(ROOT / config["input_path"])
    print(f"Loaded {len(candidate_rows)} candidate rows")

    model, tokenizer = load_inference_model(
        model_name=resolve_local_reference(ROOT, config["judge_model_name"]),
        runtime_config=config["runtime"],
        adapter_path=resolve_local_reference(ROOT, config.get("judge_adapter_path")),
    )

    prompts = [
        format_quality_prompt(
            instruction=row["instruction"],
            response=row["response"],
        )
        for row in candidate_rows
    ]
    judgments = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=config["generation"],
        max_input_length=1024,
    )

    curated_rows = []
    rejected_rows = []
    for row, judgment in zip(candidate_rows, judgments):
        scored_row = dict(row)
        scored_row["judge_output"] = judgment
        scored_row["score"] = extract_score(judgment)
        scored_row["reason"] = extract_reason(judgment)

        if scored_row["score"] is not None and scored_row["score"] >= config["score_threshold"]:
            curated_rows.append(scored_row)
        else:
            rejected_rows.append(scored_row)

    min_curated_size = config.get("min_curated_size", 0)
    fallback_applied = False
    if len(curated_rows) < min_curated_size:
        combined_rows = curated_rows + rejected_rows
        scored_rows = [row for row in combined_rows if row.get("score") is not None]
        if scored_rows:
            fallback_applied = True
            top_rows = sorted(
                scored_rows,
                key=lambda row: (row["score"], len(row.get("response", ""))),
                reverse=True,
            )[: min(min_curated_size, len(scored_rows))]
            selected_ids = {row["id"] for row in top_rows}
            curated_rows = [row for row in combined_rows if row["id"] in selected_ids]
            rejected_rows = [row for row in combined_rows if row["id"] not in selected_ids]

    curated_path = ROOT / config["output_curated_path"]
    rejected_path = ROOT / config["output_rejected_path"]
    write_jsonl(curated_path, curated_rows)
    write_jsonl(rejected_path, rejected_rows)
    write_json(
        curated_path.parent / "runtime_summary.json",
        runtime_summary(config["runtime"].get("preferred_device")),
    )
    write_json(
        curated_path.parent / "selection_metadata.json",
        {
            "score_threshold": config["score_threshold"],
            "min_curated_size": min_curated_size,
            "fallback_applied": fallback_applied,
            "curated_size": len(curated_rows),
            "rejected_size": len(rejected_rows),
        },
    )
    dump_examples(
        curated_path.parent / "high_quality_examples.json",
        curated_rows[: config["show_examples"]],
    )
    dump_examples(
        curated_path.parent / "low_quality_examples.json",
        rejected_rows[: config["show_examples"]],
    )

    print(f"Curated rows: {len(curated_rows)}")
    print(f"Rejected rows: {len(rejected_rows)}")

    print(f"Printing {config['show_examples']} high-quality examples")
    for example in curated_rows[: config["show_examples"]]:
        print("=" * 80)
        print(f"Score: {example['score']}")
        print(f"Instruction: {example['instruction']}")
        print(f"Response: {example['response'][:500]}")
        print(f"Reason: {example['reason']}")

    print(f"Printing {config['show_examples']} low-quality examples")
    for example in rejected_rows[: config["show_examples"]]:
        print("=" * 80)
        print(f"Score: {example['score']}")
        print(f"Instruction: {example['instruction']}")
        print(f"Response: {example['response'][:500]}")
        print(f"Reason: {example['reason']}")

    if config.get("push_to_hub") and config.get("hub_dataset_id"):
        to_hf_dataset(curated_rows).push_to_hub(
            config["hub_dataset_id"],
            private=config.get("hub_private", False),
        )
        print(f"Pushed curated dataset to {config['hub_dataset_id']}")


if __name__ == "__main__":
    main()
