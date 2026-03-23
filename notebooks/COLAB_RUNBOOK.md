# Colab Runbook

Use this as the blueprint for your final Colab notebook.

## Cell 1: Environment Setup

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone <YOUR_REPO_URL>
%cd /content/assignment3/assignment3
!pip install -U pip
!pip install -e .
```

If you are not using git in Colab, upload the project folder to Drive and `cd` into it directly.

## Cell 2: Hugging Face Login

```python
from huggingface_hub import login
login()
```

Accept the access terms for `GAIR/lima` on Hugging Face before running the augmentation step.

## Cell 3: Train Backward Model

```bash
!python scripts/train_backward.py --config configs/backward_train.json
```

After training, paste the Hugging Face model URL into the assignment notebook.

## Cell 4: Generate Synthetic Instructions

```bash
!python scripts/generate_instructions.py --config configs/augmentation.json
```

Copy 5 printed examples into the notebook output section if needed.

## Cell 5: Curate The Dataset

```bash
!python scripts/curate_dataset.py --config configs/curation.json
```

Copy 5 high-quality and 5 low-quality printed examples into the notebook output section if needed.

## Cell 6: Train The Final Model

```bash
!python scripts/train_forward.py --config configs/forward_train.json
```

Copy 5 printed model responses into the notebook output section if needed.

If you are uploading the server-generated artifacts instead of retraining in Colab, use:

- `artifacts/backward_model/` for the backward adapter
- `artifacts/curation/curated_pairs.jsonl` for the curated dataset
- `artifacts/final_model_selected/` for the recommended final adapter

## Cell 7: Submission Links

Add these items at the end of the notebook:

- backward model Hugging Face URL
- curated dataset Hugging Face URL
- final model Hugging Face URL
- Colab notebook share URL
