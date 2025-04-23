# Deepfake Detection Project

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/Library-TensorFlow-orange)
![GitHub](https://img.shields.io/badge/Library-OpenCV-green)

A Jupyter Notebook-based project to detect deepfake videos/images using deep learning techniques.

## Overview
This repository contains the code for a deepfake detection model, implemented in `deep_fake_projectcode.ipynb`. The notebook includes:
- Data preprocessing
- Model architecture (e.g., CNN, LSTM, or Vision Transformers)
- Training/evaluation pipelines
- Visualization of results

## Quick Start

### Prerequisites
- Python 3.8+
- Git
- pip
- Jupyter Lab
  
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Alekya2305/deepfake-detection.git
   cd deepfake-detection
   ```

2. **Install dependencies**:
 ```bash
pip install -r requirements.txt
```

3. **Launch jupyter notebook**:
  ```bash
jupyter notebook deep_fake_projectcode.ipynb
```

## 📂 Dataset
Name: [CELEB-DF]

Type: Videos

Size: [9.45 GB]

Classes:

Real

Fake

### Dataset Download
Due to size (9.45 GB), download from:  
[Google Drive Link](https://drive.google.com/drive/folders/1EDQZryUkcMjvtIiQvPCcT-RDwfAH0OdN?usp=sharing)

## Key Implementation Details  
- **Frame Sampling**: Fixed 30-frame stride ([VideoFrameSampler.py](<GitHub_link_to_file>#L12)).  
- **Temporal Pooling**: Mean aggregation across features ([model.py](<GitHub_link_to_file>#L45)).  
- **Training**: Adam (lr=0.0001), BCELoss ([train.py](<GitHub_link_to_file>#L88)).  
