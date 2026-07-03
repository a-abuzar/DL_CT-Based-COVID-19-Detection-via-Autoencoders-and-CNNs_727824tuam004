import os
import numpy as np
from PIL import Image

def create_mock_dataset(base_dir="../data", num_samples_per_class=20, img_size=(224, 224)):
    """Creates a mock dataset of random noise images to test the pipeline."""
    classes = ['COVID', 'non-COVID']
    
    for cls in classes:
        dir_path = os.path.join(base_dir, cls)
        os.makedirs(dir_path, exist_ok=True)
        
        for i in range(num_samples_per_class):
            # Generate random grayscale image
            img_array = np.random.randint(0, 256, size=(*img_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array).convert('L') # Convert to grayscale
            img.save(os.path.join(dir_path, f"mock_{cls}_{i:03d}.png"))
            
    print(f"Mock dataset created at {base_dir} with {num_samples_per_class} images per class.")

if __name__ == "__main__":
    # Ensure it creates data in the right place assuming it's run from src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    create_mock_dataset(data_dir)
