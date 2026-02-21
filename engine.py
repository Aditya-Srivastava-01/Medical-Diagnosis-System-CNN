import torch
torch.set_num_threads(1) # Limits CPU usage to prevent crashes
import torch.nn as nn
import timm
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class DiagnosisEngine:
    def __init__(self, model_path):
        self.device = torch.device("cpu") # Streamlit Cloud uses CPU
        # 1. Initialize EfficientNet-B4
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
        # 2. Load the weights we trained in Colab
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # 3. Target the last convolutional layer for Grad-CAM
        self.target_layers = [self.model.conv_head]

    def predict(self, img_rgb):
        # Resize to match training (512px)
        img_512 = cv2.resize(img_rgb, (512, 512))
        
        # Preprocessing (Normalization)
        img_input = (img_512 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(img_input).permute(2, 0, 1).float().unsqueeze(0)

        # Inference
        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor)).item()
        
        # Using the Optimal Clinical Threshold we found (0.62)
        diagnosis = "PNEUMOTHORAX DETECTED" if prob > 0.62 else "NORMAL / HEALTHY"
        
        # Grad-CAM Heatmap
        cam = GradCAM(model=self.model, target_layers=self.target_layers)
        grayscale_cam = cam(input_tensor=tensor)[0, :]
        visualization = show_cam_on_image(img_512 / 255.0, grayscale_cam, use_rgb=True)
        
        return diagnosis, prob, visualization