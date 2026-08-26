# E-Waste Classification System

An end-to-end Deep Learning project to automate the classification of electronic waste (E-Waste) into 10 categories using Transfer Learning with EfficientNetB0. The project includes real-time webcam inference inside Google Colab.

---

## Features
* **Dataset Automation:** Downloads directly via `kagglehub` API.
* **Transfer Learning:** Pretrained **EfficientNetB0** (ImageNet weights) with custom classification heads.
* **Performance:** **97.67% Test Accuracy** across 10 distinct e-waste classes.
* **Live Webcam Capture:** Real-time image capture and inference interface embedded directly in the notebook.

---

## Dataset Overview
The dataset is retrieved from Kaggle (`akshat103/e-waste-image-dataset`):
* **Total Images:** 3,000 images
* **Classes (10 categories):** Batterys, CD-ROMs, Cellphones, keyboards, Monitors, Mousses, PCB, printer-catridge, Tablets, Televisions
* **Splits:**
  * **Train:** 2,400 images
  * **Validation:** 300 images
  * **Test:** 300 images
* **Image Dimensions:** Resized to 224 × 224 pixels

---

## Model Architecture
* **Base Model:** Pretrained `EfficientNetB0` (`include_top=False`, weights frozen)
* **Pooling:** `GlobalAveragePooling2D`
* **Regularization:** `Dropout(0.2)`
* **Dense Layer:** 128 units (`ReLU` activation)
* **Output Layer:** 10 units (`Softmax` activation)
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy

---

## Performance Results

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | ~96.67% - 97.33% |
| **Overall Test Accuracy** | **97.67%** |
| **Test Loss** | 0.0386 |

---

## How to Run

### 1. Open in Google Colab
Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/) and enable a GPU accelerator (`Runtime > Change runtime type > T4 GPU`).

### 2. Install Required Dependencies
Ensure the necessary libraries are installed:
```bash
pip install tensorflow matplotlib pandas kagglehub opencv-python scikit-learn
