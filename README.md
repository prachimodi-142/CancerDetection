# CancerDetection

This repository contains a deep learning project focused on detecting cancer metastasis in histopathological images. The project utilizes advanced neural network architectures like ResNet-50 and Group Equivariant Convolutional Networks (G-CNN) with Active Learning techniques to accurately classify and identify cancerous tissues. The primary objective is to aid in the early detection of cancer, which can significantly improve treatment outcomes.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Cancer detection, especially in its early stages, is critical for effective treatment and patient survival. This project leverages deep learning techniques to analyze histopathological images for detecting cancer metastasis. By employing state-of-the-art neural networks and Active Learning, the model is designed to improve detection accuracy and adapt to new data with minimal human intervention.

## Features

- **Deep Learning Architecture**: Utilizes ResNet-50 and G-CNNs for high accuracy in image classification.
- **Active Learning**: Integrates Active Learning to iteratively improve the model with new, unlabeled data.
- **High Precision**: Aims to reduce false positives and false negatives in cancer detection.
- **Scalable Model**: The model is designed to scale with additional data and can be fine-tuned for different types of cancer.

## Technologies Used

- **Python**: The primary programming language used for the project.
- **PyTorch**: For building and training the deep learning models.
- **ResNet-50**: A deep residual network used for image classification.
- **Group Equivariant Convolutional Networks (G-CNN)**: Enhances model performance by leveraging symmetries in the data.
- **Active Learning**: Allows the model to improve over time by actively selecting the most informative samples for labeling.

## Project Structure

- `models/`: Contains the deep learning models, including ResNet-50 and G-CNN implementations.
- `data/`: Includes scripts for data preprocessing and augmentation.
- `training/`: Contains the scripts and configurations for model training and evaluation.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model experimentation.
- `utils/`: Helper functions and utilities for data handling, model evaluation, etc.
- `results/`: Stores the output results, including model predictions and performance metrics.
- `docs/`: Documentation related to the project, including methodology and findings.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prachimodi-142/CancerDetection.git
   ```
2. **Set up the environment**:
   - Create a virtual environment:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
3. **Prepare the data**:
   - Place your dataset in the `data/` directory and ensure it's properly structured according to the scripts.

4. **Train the model**:
   - Run the training script:
     ```bash
     python training/train.py
     ```

5. **Evaluate the model**:
   - Use the evaluation script to assess model performance:
     ```bash
     python training/evaluate.py
     ```

## Usage

1. **Data Preprocessing**:
   - Ensure your histopathological images are preprocessed and augmented using the provided scripts.

2. **Model Training**:
   - Train the model on your dataset using the `train.py` script. Adjust hyperparameters as needed.

3. **Model Evaluation**:
   - Evaluate the model's performance on the test set using the `evaluate.py` script.

4. **Inference**:
   - Use the trained model to perform inference on new images to detect cancer metastasis.
