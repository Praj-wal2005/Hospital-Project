import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- ARCHITECTURE FOR HEAD DIAGNOSTICS ---
class ClinicalEnsemble(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(ClinicalEnsemble, self).__init__()
        
        # Expert 1: EfficientNet (Great for detail like fractures)
        self.expert_1 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.expert_1.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes)
        )
        
        # Expert 2: ResNet50 (Deeper, better for complex masses/tumors)
        self.expert_2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.expert_2.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        out1 = self.expert_1(x)
        out2 = self.expert_2(x)
        return (out1 + out2) / 2.0
    
    def get_target_layer(self):
        # Target the last convolutional layer for the Heatmap
        return self.expert_1.features[-1]

class AIDiagnostician:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def predict_uncertainty(self, image_tensor, num_samples=20):
        self.model.train() # Enable Monte Carlo Dropout
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.model(image_tensor.to(self.device))
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0) # Average of 20 predictions
        uncertainty = np.var(predictions, axis=0) # Variance
        return mean_pred, uncertainty

    def explain_decision(self, image_tensor, original_image, target_class):
        self.model.eval()
        target_layers = [self.model.get_target_layer()]
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        # Focus the heatmap on the specific disease detected
        targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = cam(input_tensor=image_tensor.to(self.device), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        return visualization