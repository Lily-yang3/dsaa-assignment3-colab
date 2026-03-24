# Assignment 3 Submission Repository

In this repository, I organize my implementation of the Assignment 3 pipeline for
Self-Alignment with Instruction Backtranslation using QLoRA on `Qwen/Qwen3-1.7B`.
The repository includes the source code, configuration files, Colab notebook, and
submission-relevant generated artifacts.

## Included Components

- `configs/`: configuration files for backward training, augmentation, curation,
  and final instruction tuning
- `scripts/`: runnable entry points for each stage of the pipeline
- `src/assignment3/`: core implementation code
- `notebooks/assignment3_colab.ipynb`: Colab notebook used to run the pipeline
- `artifacts/backward_model/`: trained backward LoRA adapter and metadata
- `artifacts/augmentation/`: generated instruction-response candidate pairs
- `artifacts/curation/`: curated dataset and example selections
- `artifacts/final_model_best/`: final instruction-tuned LoRA adapter and demo outputs

## Python Environment

I use Python `3.10+`.

The project dependencies are declared in `pyproject.toml`, and the main libraries are:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `peft`
- `bitsandbytes`
- `huggingface_hub`
- `safetensors`
- `sentencepiece`

## Installation

I install the project in editable mode:

```bash
pip install -U pip
pip install -e .
```

If `torch` is not already available in the environment, I install a CUDA-compatible
PyTorch build first and then run the commands above.

## Running The Pipeline

To run the full pipeline from scratch, I execute the following stages in order:

```bash
python scripts/train_backward.py --config configs/backward_train.json
python scripts/generate_instructions.py --config configs/augmentation.json
python scripts/curate_dataset.py --config configs/curation.json
python scripts/train_forward.py --config configs/forward_train.json
```

## Colab Usage

To run this project in Colab, I:

1. clone the repository into `/content` or Google Drive
2. install dependencies with `pip install -e .`
3. open `notebooks/assignment3_colab.ipynb`
4. run the notebook cells in order

The default setup is designed for low-memory GPUs:

- 4-bit loading is enabled
- LoRA is used instead of full fine-tuning
- gradient checkpointing is enabled
- batch sizes are conservative

## Notes On Artifacts

This repository keeps the submission-relevant outputs under `artifacts/`.
Large downloaded base-model cache files and temporary runtime caches are intentionally
excluded from version control.
