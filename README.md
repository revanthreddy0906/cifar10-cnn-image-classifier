# CIFAR-10 CNN Image Classifier

A Convolutional Neural Network (CNN) built from scratch using TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 categories.

## 🚀 Project Overview

This project demonstrates:
- CNN architecture design
- Training and evaluation on CIFAR-10
- Overfitting analysis
- Confusion matrix visualization
- Clean ML project structure

The model is trained without transfer learning to focus on core deep learning concepts.

## 🧠 Dataset

**CIFAR-10**
- 60,000 color images (32×32)
- 10 classes:
  - airplane, automobile, bird, cat, deer
  - dog, frog, horse, ship, truck
- 50,000 training images
- 10,000 test images

## 🏗️ Model Architecture

- Convolutional blocks with ReLU
- MaxPooling layers
- Fully connected classifier
- Dropout for regularization
- Softmax applied via loss (`from_logits=True`)

Total parameters: ~1.3M

## 📊 Evaluation

Model performance is evaluated using:
- Accuracy
- Validation accuracy
- Confusion Matrix (class-wise analysis)

Confusion matrix helps identify systematic misclassifications between similar classes.

## 🛠️ Installation

```bash
git clone https://github.com/revanthreddy0906/cifar10-cnn-image-classifier.git
cd cifar10-cnn-image-classifier
pip install -r requirements.txt
```

**📈 Future Improvements**
--------------------------

-   Data Augmentation

-   Deeper CNN blocks

-   Batch Normalization

-   Transfer Learning (ResNet, VGG)

-   Per-class accuracy analysis

**📌 Key Learnings**
--------------------

-   CNNs outperform dense networks for images

-   Overfitting detection via validation curves

-   Importance of data augmentation

-   Confusion matrix as a diagnostic tool