# cross-domain-ship-detection
Reproducible implementation and experiments for cross-domain ship detection using HRSC2016, SSDD, and YOLO-based models.
# Cross-Domain Ship Detection with Optical and SAR Imagery

## 📌 Description
This project implements a cross-domain ship detection framework that leverages both **optical** and **synthetic aperture radar (SAR)** imagery.  
The approach aims to improve generalization across diverse maritime scenarios by combining datasets from multiple domains, resolutions, and imaging conditions.  
The repository includes reproducible code, pre-processing workflows, and experiment configurations.

## ✨ Key Features
- Supports both optical and SAR image modalities  
- Cross-domain training for improved detection robustness  
- Configurable data pipelines for dataset fusion  
- Reproducible experiment setup with fixed seeds and documented hyperparameters  
- Evaluation scripts with performance metrics and error analysis  
- LaTeX templates for academic reporting

## 🛠 Requirements
- Python 3.8 or higher  
- PyTorch 1.10 or higher  
- CUDA-capable GPU (RTX-class recommended)  
- Linux or Windows OS  


## 🚀 Installation

python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
