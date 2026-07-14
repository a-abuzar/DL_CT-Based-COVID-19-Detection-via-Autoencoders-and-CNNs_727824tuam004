import os
import glob
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from model import HybridCovidModel

def generate_ood_images(num_images=50):
    """Generate Out-Of-Distribution (OOD) images (noise, solid colors, gradients)"""
    images = []
    for _ in range(num_images // 3):
        # Random uniform noise
        noise = torch.rand(1, 224, 224)
        images.append(noise)
        
        # Solid colors
        val = random.random()
        solid = torch.full((1, 224, 224), val)
        images.append(solid)
        
        # Simple gradients
        grad = torch.linspace(0, 1, 224).repeat(224, 1).unsqueeze(0)
        if random.random() > 0.5:
            grad = grad.transpose(1, 2)
        images.append(grad)
        
    return images

def main():
    print("Initializing professional threshold calibration...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCovidModel()
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded.")
    else:
        print("Warning: best_model.pth not found. Using untrained weights.")
        
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 1. Evaluate In-Distribution (CT Scans)
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    ct_images = glob.glob(os.path.join(data_dir, '**/*.png'), recursive=True) + \
                glob.glob(os.path.join(data_dir, '**/*.jpg'), recursive=True)
                
    if not ct_images:
        print("No CT images found in data/ directory. Please run create_mock_data.py first.")
        return
        
    # Sample up to 200 images for speed
    sample_size = min(200, len(ct_images))
    sampled_ct = random.sample(ct_images, sample_size)
    
    print(f"\nEvaluating {sample_size} valid CT scans...")
    ct_mses = []
    with torch.no_grad():
        for img_path in sampled_ct:
            try:
                img = Image.open(img_path).convert('L')
                img_tensor = transform(img).unsqueeze(0).to(device)
                _, decoded = model(img_tensor)
                mse = torch.nn.functional.mse_loss(decoded, img_tensor).item()
                ct_mses.append(mse)
            except Exception as e:
                pass
                
    max_ct_mse = np.max(ct_mses)
    mean_ct_mse = np.mean(ct_mses)
    std_ct_mse = np.std(ct_mses)
    print(f"CT Scans MSE -> Mean: {mean_ct_mse:.4f} | Max: {max_ct_mse:.4f} | Std: {std_ct_mse:.4f}")
    
    # 2. Evaluate Out-of-Distribution (OOD)
    print(f"\nEvaluating synthetic Out-of-Distribution (non-CT) images...")
    ood_images = generate_ood_images(50)
    ood_mses = []
    
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    with torch.no_grad():
        for img_tensor in ood_images:
            img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
            _, decoded = model(img_tensor)
            mse = torch.nn.functional.mse_loss(decoded, img_tensor).item()
            ood_mses.append(mse)
            
    min_ood_mse = np.min(ood_mses)
    mean_ood_mse = np.mean(ood_mses)
    print(f"OOD Images MSE -> Mean: {mean_ood_mse:.4f} | Min: {min_ood_mse:.4f}")
    
    # 3. Calculate Optimal Threshold
    # We want a threshold that is safely above the max CT MSE, but below OOD MSEs.
    # We use max + 20% margin to safely allow all CT scans
    recommended_threshold = max_ct_mse * 1.2
    
    print(f"\n✅ Calibration Complete!")
    print(f"Recommended Threshold: {recommended_threshold:.4f}")
    
    # 4. Automatically update app.py
    app_path = os.path.join(os.path.dirname(__file__), '../app.py')
    with open(app_path, 'r') as f:
        app_code = f.read()
        
    import re
    # Replace RECONSTRUCTION_THRESHOLD = <value> with the new one
    new_app_code = re.sub(
        r"RECONSTRUCTION_THRESHOLD\s*=\s*[0-9.]+", 
        f"RECONSTRUCTION_THRESHOLD = {recommended_threshold:.4f}", 
        app_code
    )
    
    with open(app_path, 'w') as f:
        f.write(new_app_code)
        
    print(f"Successfully updated threshold in app.py to {recommended_threshold:.4f}")

if __name__ == "__main__":
    main()
