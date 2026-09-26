# AUF Dataset & MCRF Framework

This repository contains the code and data for **"The Nexus of Logos: Cost-Effective Multimodal Classification via Distilled Reasoning for Mobile User Feedback"**.

## Overview

We present:

- **AUF Dataset**: The first manually annotated, high-quality multimodal Android User Feedback dataset (3,750 entries)
- **MCRF**: Vision-Language Model-based Multimodal Classification Reasoning Framework - a teacher-student architecture for cost-effective and interpretable classification

## Dataset

The AUF dataset contains multimodal user feedback collected from the Google Android Help Community, with 21 fine-grained categories organized into 9 groups:

| Group                                                       | Categories                                              |
| ----------------------------------------------------------- | ------------------------------------------------------- |
| Accessibility / Camera                                      | Accessibility, Camera                                   |
| Wi-Fi Connectivity / Bluetooth                              | Wi-Fi Connectivity, Bluetooth                           |
| Privacy & Permissions / Malware / Security                  | Privacy & Permissions, Malware, Security                |
| Google Play Services                                        | Google Play Services                                    |
| Backup & Restore / Settings / Notifications                 | Backup & Restore, Settings, Notifications               |
| Find My Device / GPS / Location                             | Find My Device, GPS, Location                           |
| Device Protection / Forgot PIN, Pattern, Passcode / Syncing | Device Protection, Forgot PIN/Pattern/Passcode, Syncing |
| Battery / Performance / Stability                           | Battery, Performance, Stability                         |
| Other                                                       | Other                                                   |

## Project Structure

```
.
├── data/
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
