# PneumoMamba  
**PneumoMamba: A novel Mamba and CNN Dual-Path Network for Computer-Aided Diagnosis of Pneumonia using Omnidirectional Feature Extraction and Multi-Scale Asymmetric Convolution**

This repository contains the official project page for **PneumoMamba**, a novel deep learning framework for computer-aided diagnosis of pneumonia from chest X-ray (CXR) images.  
The full implementation will be released after the associated paper is accepted.

---

## 🔬 Overview

**PneumoMamba** is a dual-path neural network that integrates:

- **Mamba (Selective State Space Model)** for long-range dependency modeling  
- **Convolutional Neural Networks (CNNs)** for fine-grained local feature extraction  

The model is designed to overcome the limitations of traditional CNNs (limited global context) and Transformers (high computational cost), providing an efficient and accurate framework for pneumonia classification.

PneumoMamba achieves **state-of-the-art performance** on multi-class pneumonia classification, including:
- COVID-19
- Viral Pneumonia
- Lung Opacity
- Normal

On the Pneumonia-CXR dataset, PneumoMamba reaches a **test accuracy of 94.94%**, outperforming CNN-, Transformer-, and Mamba-based baselines.

---

## 🧠 Key Innovations

PneumoMamba introduces several novel architectural components:

### 1. Conv–OSS Dual-Path Block
A lightweight dual-branch module that:
- Uses CNNs to extract local texture and edge features  
- Uses Mamba-based OSS modules to model global spatial dependencies  

This enables simultaneous learning of **local details and global context**.

---

### 2. Omnidirectional Selective Scan Module (OSSM)
An advanced Mamba-based visual module that includes:
- **8DScan**: Scans feature maps from **8 directions** (horizontal, vertical, and diagonals)
- **Selective State Space (S6)** for adaptive long-range modeling
- **Multi-Scale Asymmetric Convolution (MSAConv)** for enriched spatial representation

This allows PneumoMamba to capture pneumonia patterns from **all spatial orientations**.

---

### 3. Focused Feature Module (FFM)
A dual-attention refinement block that integrates:
- Channel Attention
- Spatial Attention

FFM highlights subtle pneumonia-related regions and suppresses irrelevant background, improving both **accuracy and interpretability**.

---

## 📊 Experimental Results

PneumoMamba was evaluated on a large-scale chest X-ray dataset containing:

- Viral Pneumonia  
- COVID-19  
- Lung Opacity  
- Normal  

### Main Results

| Model | Test Accuracy (%) |
|------|------------------|
| ConvNeXt V2 | 93.60 |
| ViT | 88.26 |
| DeiT | 89.18 |
| PVT V2 | 93.29 |
| VMamba | 93.20 |
| MedMamba | 93.60 |
| **PneumoMamba (Ours)** | **94.94** |

The proposed method consistently outperforms CNN, Transformer, and Mamba-based baselines.

---

## 🧪 Generalization Ability

PneumoMamba was also evaluated on an external dataset (W. Ning et al.) that was not used during training:

| Metric | Score (%) |
|--------|-----------|
| Accuracy | 94.21 |
| F1-score | 94.20 |
| Sensitivity | 94.21 |
| Precision | 94.21 |

These results demonstrate **strong robustness and generalization**.

---

## 🧩 Model Interpretability

We employ **LayerCAM** to visualize model attention.  
PneumoMamba produces highly localized and clinically meaningful activation maps for different pneumonia types, accurately highlighting:

- COVID-19 ground-glass opacities  
- Viral pneumonia lesions  
- Lung opacity regions  

This provides important interpretability for clinical decision support.

---

## 📦 Code Availability

This repository currently serves as the **official project page** for PneumoMamba.

> 🚧 **The full source code, pretrained models, and training scripts will be released after the paper is officially accepted.**

This ensures compliance with journal and conference submission policies.

---

## 📌 Citation

If you find this work useful, please cite our paper:

