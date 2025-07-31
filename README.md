# Synthetic Image Generation for Enhancing Polyp Classification

This repository contains the implementation for a project focused on improving polyp classification by generating synthetic medical images using diffusion models. The goal is to mitigate data scarcity and class imbalance issues by augmenting existing datasets with realistic synthetic samples.

A detailed report explaining the full process and results can be found in [`polyp_diffusion_tfg_2025.pdf`](./polyp_diffusion_tfg_2025.pdf).


## Project Overview

This work tackles the challenges of medical image classification by:
- Developing a **baseline classifier** (EfficientNet-B0) for polyp classification.
- Training a **diffusion model** (from scratch and via fine-tuning) to generate synthetic polyp images.
- Evaluating the impact of generated images by integrating them into the classifier's training set.

The synthetic image quality is assessed by the improvement (or degradation) in classifier performance when augmented with generated samples. 

## Classifier Model

All scripts related to the baseline classifier are located in `/classifier_model`. This includes:
- Data preprocessing.
- Model training and evaluation.
- Experiments with class imbalance mitigation techniques and input image resolutions.

## Diffusion Model

The `/generator_model` directory contains:
- A scratch-trained diffusion model.
- Fine-tuning scripts using pretrained diffusion checkpoints.
- Evaluation pipeline for testing synthetic images by passing them through the classification pipeline.

## Data Analysis

The `data_analysis.ipynb` notebook explores:
- Class distributions across training, validation, and test splits.
- Dataset characteristics and imbalances.

## Utilities

The `utils.py` module includes:
- Functions for plotting training loss curves.
- Visualization tools to overlay masks on endoscopic images and display cropped regions.

## Dependencies

You can find the required libraries and versions in `requirements.txt` (add this if not yet included). Major dependencies include:
- PyTorch / torchvision
- diffusers (if using HuggingFace models)
- matplotlib, seaborn, numpy, etc.
