# Project Proposal

**Title:** COVID-19 Detection from CT Scans Using Autoencoder-Based Anomaly Detection and CNN Classification
**Student Name:** Abuzar A
**Roll Number:** 727824TUAM004

## Objective
The primary objective of this project is to construct an automated deep learning diagnostic system capable of screening COVID-19 infections from chest CT scans. By utilizing a hybrid architecture that combines an Autoencoder with a Convolutional Neural Network (CNN), the system aims to accurately distinguish between COVID-19 and non-COVID patients, minimizing false negatives and enhancing diagnostic efficiency.

## Dataset Source
**Source:** Kaggle "SARS-CoV-2 CT-Scan Dataset" (plameneduardo/sarscov2-ctscan-dataset)
**Description:** The dataset consists of 2,482 CT scan images, which include 1,252 COVID-19 positive scans and 1,230 non-COVID-19 (healthy or other pulmonary conditions) scans. The data is highly balanced, reducing the need for aggressive synthetic oversampling. Images are provided as grayscale matrices representing radiodensity.

## Architecture Overview
The proposed methodology leverages a hybrid deep learning approach:
1. **Autoencoder (AE):** Acts as a feature extractor and anomaly detector. The encoder compresses the high-dimensional CT images into a low-dimensional latent space, capturing the most salient structural features of the lungs. The decoder attempts to reconstruct the image.
2. **CNN Classifier (ResNet18 Backend):** The bottleneck (latent vector) produced by the autoencoder, alongside high-level convolutional features, is fed into a lightweight CNN classifier (based on the ResNet18 architecture). The classifier will output the binary decision (COVID vs. Non-COVID). 

This hybrid method ensures the model focuses on vital anatomical structures learned during reconstruction, improving generalizability.

## Expected Outcome
- A fully functional, modular deep learning pipeline built using PyTorch.
- Training and validation scripts that can run on Google Colab or local GPU.
- Model performance achieving >= 95% Accuracy and an ROC-AUC score of >= 0.97.
- A comprehensive evaluation including a confusion matrix, precision, recall, and F1-score to validate diagnostic reliability.
