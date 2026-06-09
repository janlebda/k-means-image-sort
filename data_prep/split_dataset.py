import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_base, output_base, split_ratio=0.75):
    categories = ['Cat', 'Dog']
    
    for category in categories:
        source_dir = os.path.join(source_base, category)
        if not os.path.exists(source_dir):
            print(f"Directory {source_dir} not found, skipping...")
            continue
            
        images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"No images found in {source_dir}, skipping...")
            continue

        train_imgs, test_imgs = train_test_split(images, train_size=split_ratio, random_state=42)
        
        # Define and create output directories
        for split, split_images in [('train', train_imgs), ('test', test_imgs)]:
            dest_dir = os.path.join(output_base, split, category)
            os.makedirs(dest_dir, exist_ok=True)
            
            print(f"Copying {len(split_images)} {split} images for {category}...")
            for img in split_images:
                shutil.copy2(os.path.join(source_dir, img), os.path.join(dest_dir, img))

if __name__ == "__main__":
    SOURCE_PATH = os.path.join('dataset', 'PetImages')
    OUTPUT_PATH = 'data'
    
    split_dataset(SOURCE_PATH, OUTPUT_PATH)
    print("Dataset split complete.")
