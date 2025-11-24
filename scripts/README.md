# Scripts Directory

This directory contains executable scripts for the schema linking project.

## Evaluation Script

### `evaluate.py`
Main model evaluation script for Text-to-SQL schema selection models.

**Usage:**
```bash
python scripts/evaluate.py --model-path /path/to/model --data-source val_prompt2.csv
```

**Key Options:**
- `--model-path`: Path to trained model (required)
- `--data-source`: Validation data file (required)
- `--model-type`: transformers or vllm (default: transformers)
- `--device`: auto, cpu, cuda (default: auto)
- `--batch-size`: Batch size for inference (default: 1)
- `--sample-limit`: Limit number of samples to evaluate
- `--use-lora`: Use LoRA weights
- `--lora-path`: Path to LoRA weights
- `--output-dir`: Output directory for results (default: outputs/evaluation)
- `--verbose`: Enable verbose logging

**Examples:**
```bash
# Basic evaluation
python scripts/evaluate.py \
    --model-path /mnt/d/modelscope/Qwen3-0.6B \
    --data-source val_prompt2.csv \
    --sample-limit 100

# With LoRA weights
python scripts/evaluate.py \
    --model-path /mnt/d/modelscope/Qwen3-0.6B \
    --use-lora \
    --lora-path outputs/dapo-Qwen3-0.6B \
    --data-source val_prompt2.csv

# Using VLLM for faster inference
python scripts/evaluate.py \
    --model-path /mnt/d/modelscope/Qwen3-0.6B \
    --model-type vllm \
    --data-source val_prompt2.csv \
    --batch-size 8
```

## Training Script

### `train_dapo_lora.sh`
Shell script for training DAPO model with LoRA fine-tuning.

**Usage:**
```bash
bash scripts/train_dapo_lora.sh
```

This script contains the training configuration and executes the DAPO training process with the specified hyperparameters.