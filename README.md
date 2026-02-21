# 🫁 Pneumothorax Detection AI (SIIM-ACR)

This project is a Deep Learning diagnostic tool designed to detect collapsed lungs (Pneumothorax) from DICOM Chest X-rays.

## 📊 Technical Performance
- **Overall System F1-Score:** 0.89
- **Sensitivity (Recall):** 80% (Clinical priority for screening)
- **Specificity (Healthy Precision):** 94%
- **Backbone:** EfficientNet-B4 (512px high-resolution input)

## 🛠️ Why this version is superior to the ResNet-50 baseline:
1. **Resolution:** Upgraded from 224px to 512px to capture fine visceral pleural lines.
2. **Architecture:** EfficientNet-B4 uses compound scaling for better medical feature extraction.
3. **Imbalance:** Addressed the 77% healthy class bias using **WeightedRandomSampling** and **pos_weight (3.48)** during training.
4. **Explainability:** Integrated **Grad-CAM** heatmaps to provide clinical interpretability.

## 🚀 Deployment
Deployed using **Streamlit Cloud** and **Docker**. The system processes raw DICOM metadata, handles MONOCHROME1 inversions, and applies CLAHE contrast enhancement in real-time.