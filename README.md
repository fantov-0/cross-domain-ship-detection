# cross-domain-ship-detection
Reproducible implementation and experiments for cross-domain ship detection using HRSC2016, SSDD, and YOLO-based models.
# Cross-Domain Ship Detection with Optical and SAR Imagery

# 🚢 cross-domain-ship-detection 🌏

This project focuses on cross-domain ship detection, aiming to improve model generalization across different data distributions. It provides a complete workflow including data preprocessing, model training, evaluation, and deployment, suitable for remote sensing imagery, port surveillance, and other scenarios.

---

## ✨ Features

- ** Domain Adaptation**: Utilizes transfer learning or domain adaptation methods to enhance detection performance on diverse datasets.
- ** Multiple Model Support**: Integrates popular object detection models 
- ** Flexible Data Processing**: Supports various remote sensing and surveillance data formats with built-in data augmentation and preprocessing.

---

## 🛠️ Requirements

-  Python >= 3.7
-  PyTorch
-  OpenCV
-  Other dependencies can be found in `requirements.txt`

---

## ⚡ Installation

```bash
git clone https://github.com/Fantov-0/cross-domain-ship-detection.git
cd cross-domain-ship-detection
pip install -r requirements.txt
```

---

## 🚦 Usage

1. **Prepare Data**: Organize source and target domain data in the `data/` directory as required.
2. **Configure Parameters**: Edit `config.yaml` or relevant scripts to fit your experiment settings.
3. **Train Model**:

  yolo train data="cfg/ships_mix.yaml" model=yolov8n.pt \



4. **Analyze Results**: Outputs detection results and metrics, supports visualization.

---

## 📁 Project Structure

```
cross-domain-ship-detection/
├── data/                # Dataset directory
├── models/              # Model definitions
├── scripts/             # Training and evaluation scripts
├── utils/               # Utility functions and preprocessing
├── config.yaml          # Configuration file
├── requirements.txt     # Dependency file
├── README.md            # Project description
```

---


