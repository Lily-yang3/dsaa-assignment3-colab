# Assignment 3 Pipeline

This project implements the four required stages from the assignment:

1. Train a backward model on `(output, instruction)` pairs from the Guanaco seed data.
2. Generate synthetic instructions for single-turn LIMA responses.
3. Curate the generated pairs with few-shot quality scoring on a 1-5 scale.
4. Train the final instruction-following model on the curated dataset.

The code is written for a server-first workflow but keeps Colab compatibility as a design constraint:

- It auto-selects the available CUDA device with the most free memory.
- It defaults to 4-bit QLoRA to stay friendly to low-memory GPUs.
- It uses plain prompt formatting instead of model-specific chat templates, which makes migration easier.
- Each assignment stage has a separate script and config file so the same pipeline can run on both server and Colab.

## Assignment Deliverables Covered

- Backward LoRA training on `timdettmers/openassistant-guanaco`
- 150-sample LIMA self-augmentation with 5 printed examples
- Few-shot self-curation with 5 high-quality and 5 low-quality examples
- Final LoRA instruction tuning on curated data with 5 generated responses
- Optional Hugging Face model and dataset pushes from the same scripts

## Project Layout

```text
assignment3/
├── configs/
│   ├── augmentation.json
│   ├── backward_train.json
│   ├── curation.json
│   └── forward_train.json
├── scripts/
│   ├── curate_dataset.py
│   ├── generate_instructions.py
│   ├── train_backward.py
│   └── train_forward.py
├── src/assignment3/
│   ├── __init__.py
│   ├── data.py
│   ├── inference.py
│   ├── models.py
│   ├── prompts.py
│   ├── runtime.py
│   └── training.py
└── pyproject.toml
```

## Recommended Execution Plan

### Phase 1: Environment

Install the package in editable mode:

```bash
cd assignment3
pip install -U pip
pip install -e .
```

Most GPU environments already ship with `torch`. If yours does not, install the correct CUDA-enabled PyTorch build first, then run `pip install -e .`.

The default augmentation config uses the public `dim/lima` mirror so the pipeline can run without gated-dataset authentication. If you have already accepted the official `GAIR/lima` access terms, you can switch the dataset name back to that source.

### Phase 2: Backward Model

Train the backward model:

```bash
python scripts/train_backward.py --config configs/backward_train.json
```

Expected artifact:

- `artifacts/backward_model/`

### Phase 3: Self-Augmentation

Generate 150 synthetic instructions from LIMA responses:

```bash
python scripts/generate_instructions.py --config configs/augmentation.json
```

Expected artifact:

- `artifacts/augmentation/candidate_pairs.jsonl`

### Phase 4: Self-Curation

Score and filter the synthetic pairs:

```bash
python scripts/curate_dataset.py --config configs/curation.json
```

Expected artifacts:

- `artifacts/curation/curated_pairs.jsonl`
- `artifacts/curation/rejected_pairs.jsonl`

### Phase 5: Final Instruction Tuning

Train the final model on the curated dataset:

```bash
python scripts/train_forward.py --config configs/forward_train.json
```

Expected artifact:

- `artifacts/final_model/`
- `artifacts/final_model_selected/` as the recommended submission-ready adapter exported from the best short-run checkpoint

## Architecture Notes

### 1. Runtime Layer

`runtime.py` handles:

- random seed setup
- automatic CUDA device selection
- BF16 and FP16 capability detection
- runtime metadata logging

### 2. Model Layer

`models.py` handles:

- tokenizer loading
- QLoRA-capable model loading
- LoRA adapter attachment
- optional Hugging Face pushes

### 3. Data Layer

`data.py` handles:

- config loading
- JSONL I/O
- dataset parsing
- single-turn filtering
- flexible extraction from different dataset schemas

### 4. Training Layer

`training.py` handles:

- prompt/target tokenization
- masked language-model supervision for response-only loss
- LoRA Trainer setup
- metric and sample artifact writing

### 5. Inference Layer

`inference.py` handles:

- batched generation
- prompt-to-output decoding
- response sampling used by augmentation, curation, and final demos

## Colab Migration Strategy

This repository is intentionally structured so that Colab only needs three environment changes:

1. Mount Drive or clone the repo into `/content`.
2. Install dependencies with `pip install -e .`.
3. Replace local output paths with Drive-backed paths if you want persistence across sessions.

No training logic should need rewriting for Colab.

## Practical Defaults For Low-Memory GPUs

The default configs are conservative:

- 4-bit loading enabled
- batch size `1`
- gradient accumulation enabled
- sequence length capped at `512`
- one epoch by default

If your server GPU is stronger, you can safely raise:

- `max_seq_length`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs`

## Important Submission Notes

- The default augmentation config points at the public `dim/lima` mirror for runtime reliability. In your write-up, you can note that it is being used as a direct accessible mirror of the LIMA conversation data.
- Push the backward adapter or merged model to Hugging Face after step 1.
- Push the curated dataset to Hugging Face after step 3.
- Push the final adapter or merged model to Hugging Face after step 4.
- Prepare a Colab notebook that calls the same scripts in order and includes the required example outputs and links.
