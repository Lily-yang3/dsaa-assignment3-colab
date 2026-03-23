from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_hf_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(rows)


def _strip_pair(instruction: str | None, response: str | None) -> dict[str, Any] | None:
    if not instruction or not response:
        return None
    instruction = instruction.strip()
    response = response.strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


def _from_messages(messages: list[Any]) -> dict[str, Any] | None:
    normalized: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "")).lower()
            content = str(message.get("content", "")).strip()
        else:
            continue
        if content:
            normalized.append((role, content))

    for idx, (role, content) in enumerate(normalized[:-1]):
        if role in {"user", "human"}:
            next_role, next_content = normalized[idx + 1]
            if next_role in {"assistant", "gpt", "bot"}:
                return _strip_pair(content, next_content)
    return None


def _from_conversations(conversations: Any) -> dict[str, Any] | None:
    if isinstance(conversations, list) and conversations:
        if isinstance(conversations[0], dict):
            return _from_messages(conversations)
        if len(conversations) >= 2 and all(isinstance(item, str) for item in conversations[:2]):
            return _strip_pair(conversations[0], conversations[1])
    return None


def _from_text_blob(text: str) -> dict[str, Any] | None:
    patterns = [
        re.compile(r"### Human:\s*(.*?)\s*### Assistant:\s*(.*)", re.DOTALL),
        re.compile(r"Human:\s*(.*?)\s*Assistant:\s*(.*)", re.DOTALL),
        re.compile(r"\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*)", re.DOTALL),
        re.compile(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*)", re.DOTALL),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _strip_pair(match.group(1), match.group(2))
    return None


def extract_instruction_response(example: dict[str, Any]) -> dict[str, Any] | None:
    direct_key_sets = [
        ("instruction", "output"),
        ("instruction", "response"),
        ("prompt", "completion"),
        ("input", "output"),
        ("question", "answer"),
    ]
    for prompt_key, response_key in direct_key_sets:
        if prompt_key in example and response_key in example:
            pair = _strip_pair(str(example[prompt_key]), str(example[response_key]))
            if pair is not None:
                return pair

    if "messages" in example:
        pair = _from_messages(example["messages"])
        if pair is not None:
            return pair

    if "conversations" in example:
        pair = _from_conversations(example["conversations"])
        if pair is not None:
            return pair

    if "text" in example:
        pair = _from_text_blob(str(example["text"]))
        if pair is not None:
            return pair

    return None


def is_single_turn_example(example: dict[str, Any]) -> bool:
    conversations = example.get("conversations")
    if isinstance(conversations, list):
        return len(conversations) == 2

    messages = example.get("messages")
    if isinstance(messages, list):
        roles = [
            str(message.get("role", "")).lower()
            for message in messages
            if isinstance(message, dict)
        ]
        return roles[:2] in (["user", "assistant"], ["human", "assistant"]) and len(roles) == 2

    return True


def train_val_split(
    rows: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    if val_ratio <= 0:
        return rows, []

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    if val_size >= len(shuffled):
        return shuffled, []
    return shuffled[val_size:], shuffled[:val_size]


def sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def load_seed_dataset(
    dataset_name: str,
    split: str,
    seed: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    try:
        dataset = dataset.shuffle(seed=seed)
    except Exception:
        pass
    rows: list[dict[str, Any]] = []
    for example in dataset:
        pair = extract_instruction_response(example)
        if pair is not None:
            rows.append(pair)
        if limit is not None and len(rows) >= limit:
            break

    random.Random(seed).shuffle(rows)
    return rows


def load_single_turn_lima(
    dataset_name: str,
    split: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    try:
        dataset = dataset.shuffle(seed=seed)
    except Exception:
        pass
    rows: list[dict[str, Any]] = []
    for example in dataset:
        if not is_single_turn_example(example):
            continue
        pair = extract_instruction_response(example)
        if pair is not None:
            rows.append(pair)
    return sample_rows(rows, sample_size=sample_size, seed=seed)
