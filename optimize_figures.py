from PIL import Image
import os
from pathlib import Path

def optimize_images(directory):
    path = Path(directory)
    for img_path in path.glob("*.png"):
        try:
            with Image.open(img_path) as img:
                # Convert to RGB to save as JPG
                rgb_img = img.convert('RGB')
                # Resize to 50% of original dimensions to massively save space
                new_size = (int(img.width * 0.5), int(img.height * 0.5))
                resized_img = rgb_img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save as JPG
                new_path = img_path.with_suffix('.jpg')
                resized_img.save(new_path, 'JPEG', quality=85)
                print(f"Compressed: {img_path.name} -> {new_path.name}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    optimize_images("paper_figures")
