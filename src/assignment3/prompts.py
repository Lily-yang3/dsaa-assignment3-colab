from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+\n", "\n", text.strip())


def format_backward_prompt(response: str) -> str:
    response = normalize_text(response)
    return (
        "You are reconstructing the most likely user instruction that could have "
        "produced the assistant response below.\n\n"
        "Write one clear, single-turn instruction.\n"
        "Do not answer the instruction.\n"
        "Do not include explanation or metadata.\n\n"
        "### Assistant Response\n"
        f"{response}\n\n"
        "### Reconstructed Instruction\n"
    )


def format_forward_prompt(instruction: str) -> str:
    instruction = normalize_text(instruction)
    return (
        "You are a helpful assistant.\n\n"
        "### Instruction\n"
        f"{instruction}\n\n"
        "### Response\n"
    )


def format_quality_prompt(instruction: str, response: str) -> str:
    instruction = normalize_text(instruction)
    response = normalize_text(response)

    few_shot = """
You are evaluating whether a candidate answer is a strong AI-assistant response to a user instruction.

Use this 5-point scale:
1 = incomplete, off-topic, vague, poorly matched to the request, or mixed with irrelevant text
2 = partially useful but only addresses the request at a high level or misses important parts
3 = helpful and mostly complete, but it reads like copied web text or a personal/forum-style response rather than an AI assistant answer
4 = strong AI-assistant answer that is clear, complete, well organized, and helpful, with only minor room for improvement
5 = excellent AI-assistant answer that is highly focused, polished, insightful, and expertly written

Few-shot examples:

Example A
Instruction: Translate "good morning" to Spanish.
Response: Buenos dias.
Score: 5
Reason: The response directly satisfies the instruction in a concise assistant style.

Example B
Instruction: What is the weather tomorrow in Tokyo?
Response: Bananas are yellow and grow in clusters.
Score: 1
Reason: The response is unrelated to the user request.

Example C
Instruction: Tell me something interesting.
Response: There are many interesting things in the world.
Score: 2
Reason: The answer is too vague and does not provide a useful response.

Example D
Instruction: Explain why eclipses happen.
Response: An eclipse happens when one celestial body moves into the shadow of another, blocking light either partially or fully.
Score: 4
Reason: The answer is clear, relevant, and complete, though it could be more detailed.
""".strip()

    return (
        f"{few_shot}\n\n"
        "Now evaluate the following candidate pair.\n\n"
        "First give a short reason for your score.\n"
        "Then write the final line exactly as:\n"
        "Score: <1-5>\n"
        "Reason: <short justification>\n\n"
        "Candidate Pair\n"
        f"Instruction: {instruction}\n"
        f"Response: {response}\n"
    )


def extract_score(text: str) -> int | None:
    score_match = re.search(r"Score:\s*([1-5])", text)
    if score_match:
        return int(score_match.group(1))

    generic_match = re.search(r"\b([1-5])\b", text)
    if generic_match:
        return int(generic_match.group(1))

    return None


def extract_reason(text: str) -> str:
    reason_match = re.search(r"Reason:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        return reason_match.group(1).strip()
    return text.strip()
