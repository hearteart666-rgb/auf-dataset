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
auf-dataset/
├── data/
│   ├── original_data/        # Original dataset files
│   │   ├── data.json         # Full dataset (3,750 entries)
│   │   ├── train.json        # Training set
│   │   └── test.json         # Test set
│   ├── distillation_data/    # Distilled reasoning outputs from teacher models
│   │   ├── distillation_data_simple_gpt-4o-mini_stage1.jsonl
│   │   ├── distillation_data_simple_gpt-4o-mini_stage2.jsonl
│   │   ├── distillation_data_simple_gemini-2.5-flash_stage1.jsonl
│   │   └── distillation_data_simple_gemini-2.5-flash_stage2.jsonl
│   └── examples/             # Image examples for demonstration
└── src/
    ├── configs/
    │   └── config.yaml       # Training configuration
    ├── data_processing/
    │   ├── category_descriptions.json    # 21 categories definitions
    │   ├── stratified_sampling.py        # Extract test set
    │   └── convert_to_training_format.py # Convert distillation to training format
    ├── distillation/
    │   └── distillation.py   # API-based knowledge distillation
    ├── inference/
    │   ├── inference.py      # Inference
    │   └── calculate_metrics.py  # Evaluation metrics calculation
    └── prompts/
        └── prompts.yaml      # Prompt templates
```

## Quick Start

### 1. Knowledge Distillation

Prepare input data for distillation:

```bash
# Copy training data as input for distillation
cp data/original_data/train.json src/distillation/input_data.json
```

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

This will generate `classification_results.json` containing the distilled reasoning from the teacher model.

### 2. Training Data Processing

Convert distillation results to training format. First, copy the distillation output:

```bash
# Copy distillation results to data processing directory
# Run conversion
cd src/data_processing
python convert_to_training_format.py
```

This script merges `classification_results.json` (from step 1) with `data/original_data/train.json` to create `training_data.json` in the format required by fine-tuning frameworks.

**Note**: Pre-computed distillation results are available in `data/distillation_data/` for reference.

### 3. Fine-tuning

Modify `src/configs/config.yaml` with your model path and dataset location, then run fine-tuning with your preferred framework (e.g., LLaMA-Factory).

### 4. Inference

Create a conda environment and install dependencies:

```bash
# Create and activate conda environment
conda create -n auf python=3.10
conda activate auf

# Install dependencies
pip install -r requirements.txt
```

Configure model path and image directory in `src/inference/inference.py`:

Run inference:

```bash
cd src/inference
python inference.py
```
