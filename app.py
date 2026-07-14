import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Ensure src is in path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import HybridCovidModel

st.set_page_config(page_title="COVID-19 CT Scanner", layout="centered")
st.title("COVID-19 Detection from CT Scans")
st.write("Upload a chest CT scan to detect the presence of COVID-19 using our Hybrid Autoencoder + ResNet18 model.")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCovidModel()
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning("Trained model weights not found! Please run train.py first. Running with untrained weights for demonstration.")
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

uploaded_file = st.file_uploader("Choose a CT Scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded CT Scan', use_column_width=True)
    
    st.write("Analyzing...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out, decoded = model(img_tensor)
        prob = torch.sigmoid(out).item()
        
        # Calculate Reconstruction Error (MSE)
        mse_loss = torch.nn.functional.mse_loss(decoded, img_tensor).item()
    
    # Threshold for filtering out non-CT scans. 
    # Calibrated value: The maximum MSE observed for valid CT scans was ~0.040
    # We use 0.048 (Max + 20% safety margin) as a data-driven boundary.
    RECONSTRUCTION_THRESHOLD = 0.048 
    
    if mse_loss > RECONSTRUCTION_THRESHOLD:
        st.warning("⚠️ **Invalid Image Detected!**")
        st.write(f"This image doesn't look like a standard CT scan. (Reconstruction Error: {mse_loss:.4f} > Threshold: {RECONSTRUCTION_THRESHOLD})")
        st.write("Please upload a valid grayscale chest CT scan.")
    else:
        # Valid CT scan, show predictions
        if prob > 0.5:
            st.error(f"**Prediction: COVID-19 Positive** (Confidence: {prob*100:.2f}%)")
        else:
            st.success(f"**Prediction: Normal / non-COVID** (Confidence: {(1-prob)*100:.2f}%)")
        
        # Optionally display the reconstruction error for debugging
        st.caption(f"Image passed validation. Reconstruction Error: {mse_loss:.4f}")
