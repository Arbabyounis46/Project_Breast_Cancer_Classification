# Project_Breast_Cancer_Classification

## Project Overview

This project aims to enhance breast cancer detection through the application of deep learning models on histopathology images. Using Convolutional Neural Networks (CNNs) and transfer learning with pre-trained architectures like VGG19 and ResNet50, the project classifies breast cancer as benign or malignant. The dataset used is the Breast Cancer Histopathological Database (BreakHis), which contains microscopic images of breast tissue.

The project demonstrates the importance of transfer learning in medical image analysis and provides insights into model performance metrics such as accuracy, precision, recall, and F1-score.

## Key Features

Utilizes pre-trained CNN models (VGG19 and ResNet50) with transfer learning.

Implements a custom CNN architecture for comparison.

Uses data augmentation and class balancing techniques to address class imbalance in the dataset.

Evaluates models on accuracy, precision, recall, F1-score, and confusion matrix.

Achieves a test accuracy of up to 97% using ResNet50.

## Technologies Used:

Python: Programming language used for the project.

TensorFlow/Keras: Deep learning framework.

OpenCV: Library for image processing.

NumPy: Library for numerical operations.

Matplotlib/Seaborn: Libraries for data visualization.

## Dataset

The project uses the Breast Cancer Histopathological Database (BreakHis), which contains over 7,000 histopathology images of breast tissue at different magnifications. https://www.kaggle.com/datasets/ambarish/breakhis.

## Model Architectures

Custom CNN: A four-layer convolutional neural network built from scratch.

VGG19: A pre-trained deep learning model from ImageNet, fine-tuned for this classification task.

ResNet50: Another pre-trained model with residual connections to handle deeper architectures effectively.

## Tool Used 

Google Colab so, not need to install any Libraries.

## Results

![image](https://github.com/user-attachments/assets/dc32f1ce-7824-4524-9736-c15a8f774f64)


## Challenges

Dataset Size: Limited dataset size impacted model generalization.

Computational Requirements: Pre-trained models require significant computational resources.

### Future Work

Expand the dataset to improve generalization.

Explore ensemble learning techniques for improved accuracy.


