# Image Captioning with LSTM and BLIP

This notebook explores deep learning approaches for the task of **image captioning**. It compares a custom LSTM-based captioning model with **BLIP (Bootstrapped Language-Image Pretraining)**, a state-of-the-art pre-trained model from Salesforce, available via Hugging Face.

---

## Overview

The project is structured around the following components:

- **Custom LSTN_Captioning Model**
  - Convolutional Neural Network (CNN) encoder for extracting image features
  - Long Short-Term Memory (LSTM) decoder for generating captions
  - Training loop and loss evaluation
- **BLIP Pretrained Model**
  - Uses `Salesforce/blip-image-captioning-base` from Hugging Face
  - Benchmarking results against the custom model
- **Performance Comparison**
  - Evaluation using example images
  - Analysis of model accuracy and linguistic quality

---

## Objectives

- Implement an end-to-end image captioning pipeline using a CNN-LSTM architecture
- Fine-tune hyperparameters and observe model behavior during training
- Compare the performance of a custom model with a cutting-edge pretrained transformer-based model

---

## Environment Setup

### Required Libraries

- torch
- torchvision
- transformers
- matplotlib
- numpy
- PIL

Install dependencies with:

```bash
pip install torch torchvision transformers matplotlib numpy pillow
