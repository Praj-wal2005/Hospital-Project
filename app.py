import os
import io
import base64
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, render_template, request
from PIL import Image

# Import the model from your pipeline file
from medical_ai_pipeline import ClinicalEnsemble, AIDiagnostician

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- KNOWLEDGE BASE: HEAD & SKULL PATHOLOGIES ---
DISEASE_INFO = {
    0: {
        "name": "Normal / Healthy",
        "severity": "Low",
        "description": "The skull vault is intact. No abnormalities visible.",
        "action": "No medical intervention required."
    },
    1: {
        "name": "Skull Fracture",
        "severity": "Critical",
        "description": "A disruption in the continuity of the skull bone is detected.",
        "action": "CRITICAL: Immediate neurosurgical consultation needed."
    },
    2: {
        "name": "Intracranial Hemorrhage",
        "severity": "Emergency",
        "description": "Abnormal high density detected inside the skull (fresh blood).",
        "action": "EMERGENCY: Immediate hospitalization required."
    },
    3: {
        "name": "Subdural Hematoma",
        "severity": "Urgent",
        "description": "Blood collection visible between the brain and the skull.",
        "action": "URGENT: Requires urgent CT/MRI confirmation."
    },
    4: {
        "name": "Tumor / Mass Effect",
        "severity": "High",
        "description": "Abnormal localized mass or shift in brain structure detected.",
        "action": "High Priority: Oncology referral needed."
    }
}

# --- 1. Load Model (5 Classes) ---
print("Loading Head CT/X-ray AI Model...")
doctor = None # Global variable for the model

try:
    model = ClinicalEnsemble(num_classes=5).to(DEVICE)
    model.eval()
    doctor = AIDiagnostician(model, DEVICE)
    print("Model Ready.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have an internet connection to download weights on the first run.")

# --- 2. Helper Functions ---
def encode_image_to_base64(image_rgb):
    """Converts a numpy image to a base64 string for HTML display"""
    img_pil = Image.fromarray(image_rgb)
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

def preprocess_image(file_stream):
    """Reads bytes from Flask upload and converts to Tensor"""
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224 for the AI Model
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    transform = A.Compose([A.Normalize(), ToTensorV2()])
    augmented = transform(image=image_resized)
    image_tensor = augmented['image'].unsqueeze(0).float()
    
    image_vis = image_resized.astype(np.float32) / 255.0
    return image_resized, image_vis, image_tensor

# --- 3. Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Safety Check: Did the model load?
    if doctor is None:
        return "System Error: AI Model failed to load. Check server terminal for details.", 500

    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # 1. Preprocess
    image_resized, image_vis, image_tensor = preprocess_image(file)

    # 2. Inference
    mean_pred, uncertainty = doctor.predict_uncertainty(image_tensor)
    
    # Get probabilities for all classes
    all_probs = mean_pred[0] 
    
    # 3. Explainability (Heatmap)
    best_class_idx = np.argmax(all_probs)
    heatmap = doctor.explain_decision(image_tensor, image_vis, target_class=best_class_idx)
    
    # 4. Prepare Data for HTML
    sorted_indices = np.argsort(all_probs)[::-1]
    
    top_diagnosis = []
    for idx in sorted_indices:
        prob_percent = round(float(all_probs[idx]) * 100, 2)
        
        # Determine Color: Red if > 50%, Green otherwise
        color_code = "#f44336" if prob_percent > 50 else "#4caf50"

        top_diagnosis.append({
            "name": DISEASE_INFO[idx]['name'],
            "prob": prob_percent,
            "severity": DISEASE_INFO[idx]['severity'],
            "color": color_code 
        })

    result = {
        "best_name": top_diagnosis[0]['name'],
        "best_prob": top_diagnosis[0]['prob'],
        "action": DISEASE_INFO[sorted_indices[0]]['action'],
        "original_img": encode_image_to_base64(image_resized),
        "heatmap_img": encode_image_to_base64(heatmap),
        "top_diagnosis": top_diagnosis
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)