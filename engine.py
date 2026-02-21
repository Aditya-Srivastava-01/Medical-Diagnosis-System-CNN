import gc
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Force single-thread to prevent memory spikes
torch.set_num_threads(1)

class DiagnosisEngine:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        # Load the B4 architecture
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.target_layers = [self.model.conv_head]

    def predict(self, img_rgb):
        # 384px is the 'Sweet Spot' for Streamlit Cloud RAM
        img_res = cv2.resize(img_rgb, (384, 384))
        
        # Preprocessing
        img_input = (img_res / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(img_input).permute(2, 0, 1).float().unsqueeze(0)

        # Inference
        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor)).item()
        
        diagnosis = "PNEUMOTHORAX DETECTED" if prob > 0.62 else "NORMAL / HEALTHY"
        
        # Grad-CAM with Memory Management
        try:
            cam = GradCAM(model=self.model, target_layers=self.target_layers)
            grayscale_cam = cam(input_tensor=tensor)[0, :]
            visualization = show_cam_on_image(img_res / 255.0, grayscale_cam, use_rgb=True)
            # Cleanup CAM immediately
            del cam
        except Exception:
            visualization = img_res

        # --- FORCE CLEANUP ---
        del tensor
        gc.collect() 
        
        return diagnosis, prob, visualization
