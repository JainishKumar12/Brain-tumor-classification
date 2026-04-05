import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gdown
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CNN — 
# ─────────────────────────────────────────────
class brain_tumor(nn.Module):
  def __init__(self):
    super (brain_tumor , self).__init__()

    self.conv_layers = nn.Sequential(
        nn.Conv2d(3 , 32 , kernel_size=3 , padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(32 , 64 , kernel_size=3 , padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64 , 128 , kernel_size=3 , padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(128 , 256 , kernel_size=3 , padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2,2)

    )

    self.fc_layers= nn.Sequential(
        nn.Flatten(),
        nn.Linear(256*14*14 , 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512 , 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128,4)
    )

  def forward(self,x):
    x = self.conv_layers(x)
    x = self.fc_layers(x)
    return x

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = "best_modeltumor.pth"
GDRIVE_FILE_ID = "1q_EBaAN-HDzZf1KLbxkrw7dU2grYCgsa"  # ⚠️ Replace with your actual file ID

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = brain_tumor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_INFO = {
    "Glioma":     ("🔴", "Malignant tumor arising from glial cells. Requires urgent medical attention."),
    "Meningioma": ("🟠", "Tumor forming on the brain's protective membranes. Often benign but needs monitoring."),
    "No Tumor":   ("🟢", "No tumor detected. Scan appears healthy."),
    "Pituitary":  ("🔵", "Tumor located in the pituitary gland. Usually benign and treatable."),
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs.numpy()

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("## 🧠 Brain Tumor Classifier")
st.markdown("Upload a brain MRI scan and the model will classify it into one of 4 categories.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            model = load_model()
            label, probs = predict(image, model)

        emoji, description = CLASS_INFO[label]
        st.markdown(f"### {emoji} {label}")
        st.markdown(f"_{description}_")
        st.markdown("---")
        st.markdown("**Confidence Scores**")
        for i, cls in enumerate(CLASS_NAMES):
            e, _ = CLASS_INFO[cls]
            st.progress(float(probs[i]), text=f"{e} {cls}: {probs[i]*100:.1f}%")

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")

else:
    st.info("👆 Upload a brain MRI image to get started.")

    st.markdown("### 📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "91.5%")
    col2.metric("Macro F1", "0.91")
    col3.metric("Test Samples", "1,600")
