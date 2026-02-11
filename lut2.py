import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def apply_style_to_folder(source_ref_path, target_ref_path, folder_path, output_folder):
    # 1. Train the model on your Reference Pair (The girl in blue)
    print("Training color model...")
    source_img = Image.open(source_ref_path).convert('RGB')
    target_img = Image.open(target_ref_path).convert('RGB')
    
    # Downsample for faster training (speed optimization)
    source_img.thumbnail((200, 200))
    target_img.thumbnail((200, 200))
    
    X = np.array(source_img).reshape(-1, 3) # Raw pixels
    y = np.array(target_img).reshape(-1, 3) # Edited pixels
    
    # Polynomial Regression learns the non-linear curve (Contrast + Saturation)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)
    
    # 2. Apply to all photos in the folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Processing {filename}...")
            img_path = os.path.join(folder_path, filename)
            
            # Load the new photo
            new_img = Image.open(img_path).convert('RGB')
            original_size = new_img.size
            pixels = np.array(new_img).reshape(-1, 3)
            
            # Predict new colors
            new_pixels = model.predict(pixels)
            new_pixels = np.clip(new_pixels, 0, 255).astype(np.uint8)
            
            # Save
            result_img = Image.fromarray(new_pixels.reshape(new_img.height, new_img.width, 3))
            result_img.save(os.path.join(output_folder, f"Edited_{filename}"))

# Usage
apply_style_to_folder("d.jpg", "l.jpg", "./Raw_Photos", "./Edited_Photos")