# 🏥 MedVision AI Pro

#Advanced Head CT/X-Ray Analysis with Explainable AI (XAI)

MedVision AI Pro is a medical diagnostic web application powered by a Deep Learning ensemble. It detects cranial pathologies from CT scans/X-rays and provides visual explainability (heatmaps) to assist medical professionals in differential diagnosis.

#  Key Features
1. Deep Learning Ensemble:Utilizes a custom `ClinicalEnsemble` architecture combining  EfficientNet-B0 (for fine detail) and ResNet50 (for complex patterns) to improve classification accuracy
2. Explainable AI (XAI):** Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) to generate attention heatmaps, visualizing exactly which regions of the brain influenced the diagnosis.
3. Uncertainty Quantification:** Implements Monte Carlo Dropout to estimate prediction confidence and uncertainty variances, ensuring clinical reliability.
4. Interactive Web Interface: A responsive Flask-based UI with real-time probability animations and severity grading.

# Diagnostic Capabilities

The model is trained to classify 5 specific conditions:
1. Normal / Healthy
2. Skull Fracture (Critical)
3. Intracranial Hemorrhage (Emergency)
4. Subdural Hematoma (Urgent)
5. Tumor / Mass Effect (High Priority)

# 🛠️ Tech Stack
Framework: PyTorch, Torchvision
Backend: Flask (Python)
Image Processing: OpenCV, Albumentations
Visualization:** Matplotlib, CSS3 Animations

# Installation
1.  Clone the repository.
2.  Install dependencies:bash
    pip install -r requirements.txt
3.  Run the application:
    bash python app.py
    
4.  Open http://localhost:5000 in your browser.
