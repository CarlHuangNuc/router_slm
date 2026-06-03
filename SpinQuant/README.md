# SpinQuant - LLM Quantization Framework

**A unified framework for quantizing Large Language Models using rotation-based optimization**

---

## 📖 Overview

SpinQuant is a quantization framework that uses learned rotation matrices to compress Large Language Models while maintaining high accuracy. This repository provides implementations for multiple model families with easy-to-use scripts.

---

## 🤖 Supported Models

This repository currently supports the following model families:

| Model Family | Status | Documentation | Scripts | Python Code |
|--------------|--------|---------------|---------|-------------|
| **Llama3** | ✅ Supported | - | `scripts/*llama3*.sh` | `models/llama3/` |
| **Qwen2** | ✅ Supported | - | `scripts/*qwen2*.sh` | `models/qwen2/` |
| **Qwen3** | ✅ Supported | [README_Qwen3.md](README_Qwen3.md) | `scripts/*qwen3*.sh` | `models/qwen3/` |
| **Grantie** | ✅ Supported | [README_Granite.md](README_Granite.md) | `scripts/*granite*.sh` | `models/granite/` |

> 💡 **For detailed usage instructions, please refer to the model-specific README files.**

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirement.txt
```

### Usage

#### Granite

```bash
# Run full pipeline
bash run_granite_quantization.sh
```

#### Qwen3

```bash
# Run full pipeline
bash run_qwen3_quantization.sh
```

For detailed instructions, see [README_Qwen3.md](README_Qwen3.md)

#### Qwen2

```bash
bash scripts/10_optimize_rotation_qwen2.sh
bash scripts/2_eval_rotation_model_qwen2.sh
```

#### Llama3

```bash
bash scripts/2_eval_rotation_model_llama3.sh
bash scripts/2_eval_rotation_llama3.sh
```

---

## 📁 Project Structure

```
SpinQuant/
├── models/                          # Model-specific implementations
│   ├── llama3/                      # Llama3 quantization code
│   ├── qwen2/                       # Qwen2 quantization code
│   └── qwen3/                       # Qwen3 quantization code
├── scripts/                         # Shell scripts for training
├── utils/                           # Utility functions
├── train_utils/                     # Training utilities
├── eval_utils/                      # Evaluation utilities
├── run_qwen3_quantization.sh        # Qwen3 quantization pipeline
├── requirement.txt                  # Python dependencies
├── README.md                        # This file
└── README_Qwen3.md                  # Detailed Qwen3 guide
```

---

## 📚 Documentation

For detailed usage instructions, please refer to the model-specific documentation:

- **[Qwen3 Guide](README_Qwen3.md)** - Complete guide for Qwen3 quantization

---
