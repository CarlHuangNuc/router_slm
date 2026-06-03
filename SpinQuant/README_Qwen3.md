# SpinQuant + GPTQ for Qwen3 

This guide provides a comprehensive walkthrough for running SpinQuant quantization on Qwen3 models.

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Full Training Pipeline](#full-training-pipeline)
- [Step-by-Step Guide](#step-by-step-guide)
- [Configuration](#configuration)
  - [LoRA Configuration](#lora-configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

SpinQuant is a quantization method that uses learned rotation matrices to improve the quantization performance of large language models. This implementation supports Qwen3 models with the following features:

- **8-bit weight quantization** with optimized rotation
- **16-bit activation and KV cache** for accuracy
- **Post-training quantization (PTQ)** support
- **Multi-GPU training** with distributed data parallel

### Supported Models
- Qwen3-0.6B
- Qwen3-1.7B

---

## Environment Setup

### 1. Core Dependencies

Install the base requirements:

The `requirement.txt` includes:
- `transformers==4.51.0` - HuggingFace transformers library
- `accelerate==0.31.0` - Distributed training support
- `datasets==2.20.0` - Dataset loading and processing
- `sentencepiece` - Tokenization
- `tensorboardX` - Training visualization
- `torch==2.1.2` - PyTorch framework

### 2. Additional Dependencies for Qwen3

For Qwen3-specific features, install:

```bash
pip install transformers==4.51.0 tokenizers tiktoken jsonlines openpyxl jinja2==3.1.0 islpy onnxscript==0.2.3 aenum
```

**Package explanations:**
- `transformers==4.51.0` - Updated version with Qwen3 support
- `tokenizers` - Fast tokenization backend
- `tiktoken` - Qwen3 tokenizer support
- `jsonlines` - Dataset format support
- `openpyxl` - Excel file handling for results
- `jinja2==3.1.0` - Template engine
- `islpy` - Integer set library for optimization
- `onnxscript==0.2.3` - ONNX export support
- `aenum` - Advanced enumerations

### 3. Optional: Weights & Biases (W&B)

For experiment tracking and visualization:

```bash
pip install wandb accelerate==0.34.0
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 5. Environment Variables (Optional)

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Set W&B mode (if not using W&B)
export WANDB_MODE=disabled

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1
```

---

## Quick Start

### Run Full Pipeline

The easiest way to run the complete SpinQuant workflow:

```bash
bash run_qwen3_quantization.sh
```

This will automatically execute all three steps:
1. ✓ Optimize rotation matrices
2. ✓ Evaluate rotation model
3. ✓ Evaluate PTQ (Post-Training Quantization)

All logs will be saved in the `logs/` directory.

---

## Full Training Pipeline

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3 SpinQuant Pipeline                 │
└─────────────────────────────────────────────────────────────┘

  Input: Qwen3-0.6B Model
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Optimize Rotation Matrices                          │
│ ─────────────────────────────────────────────────────────── │
│ • Script: scripts/10_optimize_rotation_qwen3.sh             │
│ • Python: models/qwen3/optimize_rotation_qwen3.py           │
│ • Duration: ~1-2 hours (100 steps)                          │
│ • Output: rotation/R.bin                                    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Evaluate Rotation Model                             │
│ ─────────────────────────────────────────────────────────── │
│ • Script: scripts/2_eval_rotation_model_qwen3.sh            │
│ • Python: models/qwen3/rotation_model_generation_qwen3.py   │
│ • Duration: ~30 minutes                                     │
│ • Output: no_had_model_random_H/                            │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Evaluate PTQ                                        │
│ ─────────────────────────────────────────────────────────── │
│ • Script: scripts/2_eval_ptq_qwen3.sh                       │
│ • Python: models/qwen3/ptq_qwen3.py                         │
│ • Duration: ~20 minutes                                     │
│ • Output: gptq.pth                                          │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
  Final Quantized Model
```

### Output Structure

```
output/qwen3_0.6b/
├── rotation/
│   └── R.bin                      # Optimized rotation matrix
├── no_had_model_random_H/         # Rotated model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── gptq.pth                       # Final quantized model
└── logs/                          # Training logs
```

---

## Step-by-Step Guide

### Step 1: Optimize Rotation Matrices

**Purpose:** Learn optimal rotation matrices to improve quantization quality.

**Command:**
```bash
bash scripts/10_optimize_rotation_qwen3.sh
```

**Key Parameters:**
- `--input_model`: Path to Qwen3-0.6B model
- `--output_rotation_path`: Where to save rotation matrix
- `--max_steps 100`: Number of optimization steps
- `--learning_rate 1.5`: Learning rate for rotation optimization
- `--w_bits 8`: 8-bit weight quantization
- `--a_bits 16`: 16-bit activation
- `--k_bits 16`: 16-bit key cache
- `--v_bits 16`: 16-bit value cache
- `--k_groupsize 128`: Group size for key quantization
- `--v_groupsize 128`: Group size for value quantization

**GPU Requirements:**
- 2 GPUs (configurable via `--nproc_per_node`)
- ~16GB VRAM per GPU

**Expected Output:**
```
Step 1/100: loss=0.234
Step 50/100: loss=0.156
Step 100/100: loss=0.089
Rotation matrix saved to: output/qwen3_0.6b/rotation/R.bin
```

---

### Step 2: Evaluate Rotation Model

**Purpose:** Apply learned rotation and generate rotated model.

**Command:**
```bash
bash scripts/2_eval_rotation_model_qwen3.sh
```

**Key Parameters:**
- `--input_model`: Original Qwen3 model
- `--optimized_rotation_path`: Path to R.bin from Step 1
- `--save_no_had_model_path`: Where to save rotated model
- `--rotate`: Enable rotation application
- `--do_eval True`: Run evaluation

**Expected Output:**
```
Applying rotation matrices...
Evaluating model performance...
Perplexity: 12.34
Model saved to: output/qwen3_0.6b/no_had_model_random_H/
```

---

### Step 3: Evaluate PTQ (Post-Training Quantization)

**Purpose:** Apply final quantization and evaluate performance.

**Command:**
```bash
bash scripts/2_eval_ptq_qwen3.sh
```

**Key Parameters:**
- `--input_model`: Original or rotated model
- `--save_qmodel_path`: Where to save quantized model
- `--w_bits 8`: 8-bit weight quantization
- `--w_clip`: Enable weight clipping
- `--lora_path`: Path to LoRA adapter (optional, see [LoRA Configuration](#lora-configuration))
- `--calibration_data`: Custom calibration dataset (optional)
- `--test_prompt`: Test prompt for generation (optional)

**LoRA Configuration:**

The PTQ evaluation script supports optional LoRA (Low-Rank Adaptation) fine-tuned models:

1. **Using LoRA adapter:**
   - Set `lora_path` in the script to your LoRA adapter directory
   - The script will automatically use the rotated model from Step 2
   - Example:
     ```bash
     lora_path='/path/to/your/lora_adapter'
     ```

2. **Without LoRA (default):**
   - Set `lora_path` to empty string: `lora_path=''`
   - The script will use the rotated model from Step 2 as input
   - Example:
     ```bash
     lora_path=''
     ```

**Important:** When `lora_path` is empty, the script automatically switches the input model to use the rotated model (`no_had_model_random_H`) from Step 2 instead of the original HuggingFace model.

**Expected Output:**
```
Quantizing model...
Evaluating quantized model...
Perplexity: 12.56
Quantized model saved to: output/qwen3_0.6b/gptq.pth
```

---

## Configuration

### Modify Model Path

Edit the scripts to use your model path:

```bash
# In scripts/10_optimize_rotation_qwen3.sh
Model_id='/your/path/to/Qwen3-0.6B'
output_dir='/your/output/directory'
```

### Adjust GPU Configuration

```bash
# Use 4 GPUs instead of 2
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29505 ...

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/10_optimize_rotation_qwen3.sh
```

### Modify Quantization Settings

```bash
# 4-bit weight quantization
--w_bits 4

# Different group sizes
--k_groupsize 64
--v_groupsize 64

# More training steps
--max_steps 200
```

### LoRA Configuration

#### Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method. The PTQ evaluation script (Step 3) supports optional LoRA adapters, allowing you to quantize LoRA fine-tuned models.

#### Configuration in `scripts/2_eval_ptq_qwen3.sh`

**Option 1: Using LoRA Adapter**

If you have a LoRA fine-tuned model, set the `lora_path` variable:

```bash
# Set path to your LoRA adapter directory
lora_path='/path/to/your/lora_adapter'
```

The script will:
- Load the LoRA adapter
- Apply it to the base model
- Perform quantization on the combined model

**Option 2: Without LoRA (Default)**

If you don't need LoRA, set `lora_path` to an empty string:

```bash
# Disable LoRA by setting empty string
lora_path=''
```

The script will:
- Automatically use the rotated model from Step 2 (`no_had_model_random_H`)
- Skip LoRA loading
- Perform quantization on the rotated model only

#### Important Notes

⚠️ **Critical:** The `lora_path` variable **must** be set in the script:
- **If using LoRA:** Set to the full path of your LoRA adapter
- **If not using LoRA:** Set to empty string `''` (not commented out)

📝 **Model Selection Logic:**
```bash
if [ -z "$lora_path" ]; then
    # No LoRA: Use rotated model from Step 2
    Model_id='output/qwen3_0.6b/no_had_model_random_H'
fi
```

#### Example Configurations

**Example 1: With LoRA**
```bash
# In scripts/2_eval_ptq_qwen3.sh
Model_id='/path/to/Qwen3-0.6B'
lora_path='/path/to/lora_adapter'  # LoRA enabled
calibration_dataset='/path/to/calibration_data.json'
```

**Example 2: Without LoRA**
```bash
# In scripts/2_eval_ptq_qwen3.sh
Model_id='/path/to/Qwen3-0.6B'  # Will be overridden
lora_path=''  # LoRA disabled - uses rotated model
calibration_dataset=''  # Use default wikitext
```

#### Additional PTQ Options

**Custom Calibration Dataset:**
```bash
# Use custom calibration data (JSON format)
calibration_dataset='/path/to/calibration_data.json'

# Use default wikitext dataset
calibration_dataset=''
```

**Test Prompt for Generation:**
```bash
# Test with a specific prompt
test_prompt='<|im_start|>user\n中国首都在哪里？<|im_end|>\n<|im_start|>assistant\n'

# Skip generation test
test_prompt=''
```

#### Workflow Comparison

**With LoRA:**
```
Original Model → Rotation (Step 1-2) → + LoRA Adapter → Quantization (Step 3)
```

**Without LoRA:**
```
Original Model → Rotation (Step 1-2) → Quantization (Step 3)
```

### Batch Size Tuning

```bash
# Increase batch size if you have more VRAM
--per_device_train_batch_size 2

# Decrease if OOM (Out of Memory)
--per_device_train_batch_size 1
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `--per_device_train_batch_size 1`
- Enable gradient checkpointing: `--gradient_checkpointing True`
- Use fewer GPUs or smaller model
- Clear CUDA cache: `torch.cuda.empty_cache()`

#### 2. Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
pip install transformers==4.51.0
```

#### 3. Port Already in Use

**Error:**
```
Address already in use
```

**Solution:**
Change the master port in the script:
```bash
--master_port=29506  # Use a different port
```

#### 4. Model Not Found

**Error:**
```
OSError: /path/to/model does not exist
```

**Solution:**
- Verify model path is correct
- Download model: `huggingface-cli download Qwen/Qwen3-0.6B`
- Update `Model_id` in scripts
- If using PTQ without LoRA, ensure Step 2 completed successfully and `no_had_model_random_H` exists

#### 6. LoRA Path Issues

**Error:**
```
LoRA adapter not found or invalid
```

**Solution:**
- Verify `lora_path` points to a valid LoRA adapter directory
- Check that the adapter is compatible with the base model
- If not using LoRA, ensure `lora_path=''` (empty string)
- Verify the LoRA adapter contains required files (adapter_config.json, adapter_model.bin, etc.)

#### 7. Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Make scripts executable
chmod +x full_training_qwen3.sh
chmod +x scripts/*.sh

# Check output directory permissions
mkdir -p output/qwen3_0.6b
chmod 755 output/qwen3_0.6b
```

### Monitoring Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor logs in real-time
tail -f logs/10_optimize_rotation_qwen3.log

# Check all logs
ls -lh logs/
```

### Performance Tips

1. **Use tmux/screen** for long-running jobs:
   ```bash
   tmux new -s spinquant
   bash full_training_qwen3.sh
   # Detach: Ctrl+B, then D
   ```

2. **Enable mixed precision** (already enabled with `--bf16 True`)

3. **Use gradient checkpointing** to save memory

4. **Monitor disk space** - models and logs can be large

---

## Additional Resources

### File Structure

```
SpinQuant/
├── models/
│   ├── qwen3/
│   │   ├── optimize_rotation_qwen3.py
│   │   ├── rotation_model_generation_qwen3.py
│   │   └── ptq_qwen3.py
│   ├── qwen2/
│   └── llama3/
├── scripts/
│   ├── 10_optimize_rotation_qwen3.sh
│   ├── 2_eval_rotation_model_qwen3.sh
│   └── 2_eval_ptq_qwen3.sh
├── utils/
├── train_utils/
├── eval_utils/
├── full_training_qwen3.sh
├── requirement.txt
└── README_Qwen3.md
```

### Useful Commands

```bash
# Run full pipeline
bash full_training_qwen3.sh

# Run individual steps
bash scripts/10_optimize_rotation_qwen3.sh
bash scripts/2_eval_rotation_model_qwen3.sh
bash scripts/2_eval_ptq_qwen3.sh

# Check logs
cat logs/10_optimize_rotation_qwen3.log

# Clean up
rm -rf logs/*.log
rm -rf output/qwen3_0.6b/*
```

### Contact & Support

For issues and questions:
- Check logs in `logs/` directory
- Review error messages carefully
- Verify environment setup
- Check GPU availability and memory

---

## Summary

This guide covers:
- ✅ Complete environment setup
- ✅ Full training pipeline explanation
- ✅ Step-by-step execution guide
- ✅ Configuration options
- ✅ Troubleshooting common issues

For quick start, simply run:
```bash
bash full_training_qwen3.sh
```

All logs will be saved in `logs/` directory for debugging and monitoring.
