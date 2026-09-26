# AUF Dataset & MCRF Framework

This repository contains the code and data for **"The Nexus of Logos: Cost-Effective Multimodal Classification via Distilled Reasoning for Mobile User Feedback"**.

## Overview

We present:

- **AUF Dataset**: The first manually annotated, high-quality multimodal Android User Feedback dataset (3,750 entries)
- **MCRF**: Vision-Language Model-based Multimodal Classification Reasoning Framework - a teacher-student architecture for cost-effective and interpretable classification

## Dataset

The AUF dataset contains multimodal user feedback collected from the Google Android Help Community, with 21 fine-grained categories.

## Project Structure

```
.
├── data/
│   ├── examples           # image examples for demonstration
│   ├── data.json          # Full dataset
│   ├── train.json         # Training set
│   └── test.json          # Test set
├── src/
│   ├── configs/           # Training configurations
│   ├── data_processing/   # Data preprocessing scripts
│   ├── distillation/      # Knowledge distillation from teacher models
│   ├── inference/         # Inference scripts
│   └── prompts/           # Prompt templates
└── README.md
```

## Quick Start

### 1. Knowledge Distillation

Configure your API settings in `src/distillation/.env`:

```
API_KEY=your-api-key
API_BASE_URL=your-api-base-url
MODEL=gpt-4o-mini
```

Run distillation:

```bash
cd src/distillation
python distillation.py
```

### 2. Fine-tuning

Modify `src/configs/config.yaml` with your model path and dataset location, then run fine-tuning with your preferred framework (e.g., LLaMA-Factory).

### 3. Inference

```bash
cd src/inference
python inference.py
```
