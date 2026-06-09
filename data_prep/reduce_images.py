import os
from PIL import Image

def reduce_images(target_dir, size=(128, 128)):
    """
    Reduces the dimensions of images in the target directory using Lanczos resampling
    to preserve as much variance/detail as possible.
    """
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary (handles RGBA or grayscale)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Use LANCZOS for high quality downsampling
                        img_resized = img.resize(size, Image.Resampling.LANCZOS)
                        img_resized.save(img_path, quality=85, optimize=True)
                    print(f"Reduced: {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    TRAIN_DIR = os.path.join('data', 'train')
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory {TRAIN_DIR} not found. Run split_dataset.py first.")
    else:
        print("Starting image reduction...")
        reduce_images(TRAIN_DIR)
        print("Image reduction complete.")
