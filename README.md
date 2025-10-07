# Counterfeit Drug Detection with ResNet

This project presents a deep learning pipeline for detecting counterfeit pharmaceutical products using image classification. Built around a fine-tuned ResNet architecture, the model is designed to identify subtle visual discrepancies in drug packaging, labeling, and imprint — helping healthcare providers and regulators combat counterfeit distribution.

## Overview

This project explores the use of deep learning for counterfeit drug detection through image classification. The experimental process involved curating multiple custom datasets from verified pharmaceutical sources, simulating real-world scenarios of counterfeit packaging and pill imprint variations.

To address class imbalance, I engineered synthetic datasets using a combination of:
- SMOTE for oversampling minority classes
- GenAI, and Augmentations  + OCR for realistic counterfeit image generation
- Custom scripts for stratified splitting and reproducible preprocessing

The model is built on a fine-tuned ResNet-50 architecture, trained and evaluated across these datasets to benchmark performance under varying distribution conditions. Evaluation metrics include precision, recall, F1-score, with confusion matrix analysis to assess misclassification risks in healthcare contexts.

You can explore the full pipeline and dataset engineering logic in the following repositories:
- [Dataset Curation & Augmentation Scripts](https://github.com/yourusername/counterfeit-dataset-tools)
- [GenAI Counterfeit Drug Image Generation Script](https://github.com/arabomeivan/CounterFeitDrugImageGenerator)
- [Deployment & Integration (React Native)](https://github.com/yourusername/fractionnine-app)

This project was designed with deployment in mind — from mobile-first integration to QA-driven evaluation
## Model Architecture

- Base Model: ResNet-101 (pretrained on ImageNet)
- Fine-Tuning: Last few layers retrained on domain-specific dataset
- Input: RGB images of drug packaging and pills
- Output: Multi classification — Authentic, Counterfeit, Non-Medication

## Dataset Pipeline

- Source: Curated dataset of verified authentic and counterfeit drug images
- Preprocessing:
  - Resizing and normalization
  - Label encoding and stratified splitting
  - Augmentation (rotation, blur, contrast shift)
- Class Imbalance Handling:
  - Weighted loss function
  - Synthetic oversampling (SMOTE + GAN-based augmentation)

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix Analysis

## Tools & Frameworks

- Python, PyTorch
- Pilow, Albumentations
- Scikit-learn, Matplotlib
- CI/CD pipeline for reproducible training and deployment

## Deployment

- Exported as TorchScript for mobile compatibility
- Integrated into a React Native frontend for real-time scanning

## Notable Features

- Modular training pipeline with reproducible splits
- QA-driven evaluation and logging
- Designed for edge deployment and low-latency inference
- Built with a systems mindset — from data curation to model serving


## Future Work

- Explore other techniques to pattern recognition
- Integrate blockchain-based record logging for traceability
- Explore transformer-based vision models (e.g., ViT, SAM)

## Author

**Ivan Arabome**  
Vue/Nuxt Developer | SDET | ML Practitioner  
[LinkedIn](https://www.linkedin.com/in/ivanarabome) • [Portfolio](https://your-portfolio-link.com)
