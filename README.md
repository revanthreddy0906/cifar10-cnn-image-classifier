# CIFAR-10 CNN Image Classifier

A Convolutional Neural Network (CNN) built **from scratch** using **TensorFlow/Keras** to classify images from the CIFAR-10 dataset into 10 object categories.

This project focuses on **understanding CNN design, training stability, regularization, and evaluation**, without using pretrained models or transfer learning.

---

## 🚀 Project Overview

This project demonstrates:

- CNN architecture design from first principles  
- Training and evaluation on the CIFAR-10 dataset  
- Overfitting detection and mitigation  
- Confusion matrix–based error analysis  
- Clean, modular ML project structure  

The goal is to gain **hands-on understanding of deep learning fundamentals**, rather than maximizing benchmark scores.

---

## 🧠 Dataset

**CIFAR-10**

- 60,000 color images (32×32)
- 10 classes:
  - airplane, automobile, bird, cat, deer  
  - dog, frog, horse, ship, truck
- 50,000 training images
- 10,000 test images

---

## 🏗️ Model Architecture

- **3 Convolutional blocks**
  - Conv2D → Batch Normalization → ReLU → MaxPooling
- **Classifier**
  - Dense(256) → Dropout(0.5)
  - Dense(128) → Dropout(0.3)
  - Dense(10 logits)
- Loss handled using `SparseCategoricalCrossentropy(from_logits=True)`

**Total parameters:** ~1.3M

---

## ⚙️ Training Strategy

- Input normalization
- Data augmentation (horizontal flip, rotation, zoom)
- Batch Normalization for stable training
- Dropout for regularization
- Early stopping to prevent overfitting

Training was stopped automatically once validation performance stopped improving.

---

## 📊 Evaluation & Results

- **Best validation accuracy:** ~67%
- Small train–validation gap → good generalization
- Performance analyzed using a **confusion matrix**

Key observations:
- Strong performance on classes like automobile, frog, ship, and truck
- Expected confusion between visually similar classes (cat ↔ dog, deer ↔ horse)
- Confusion matrix used as a diagnostic tool rather than accuracy alone

---

## 🛠️ Installation

```bash
git clone https://github.com/revanthreddy0906/cifar10-cnn-image-classifier.git
cd cifar10-cnn-image-classifier
pip install -r requirements.txt
```
---

**📌 Key Learnings**
--------------------

-   CNNs outperform dense networks for image data

-   Correct data pipelines are critical for stable training

-   Overfitting must be diagnosed using validation metrics

-   Confusion matrices provide deeper insight than accuracy alone

-   Regularization and early stopping are essential for generalization

* * * * *

**📈 Future Improvements**
--------------------------

-   Stronger data augmentation (MixUp / CutOut)

-   Learning rate scheduling

-   Residual connections (ResNet-style blocks)

-   Transfer learning with pretrained backbones

-   Per-class precision and recall analysis