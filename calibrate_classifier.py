import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.linear_model import LogisticRegression
import pickle

MODEL_PATH = os.path.join('model', 'model_v1.keras')
DATA_PATH = os.path.join("data", "train") 

print("=== CLASSIFIER HEAD CALIBRATION FOR MODEL V1 ===")

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Could not find: {MODEL_PATH}")
    exit()

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Target data folder does not exist: {DATA_PATH}")
    exit()

# POPRAWKA: Ładowanie modelu z dysku przed pętlą przetwarzania plików
print("Loading model_v1.keras into memory...")
model = tf.keras.models.load_model(MODEL_PATH)

subfolders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
print(f"Target directory: {DATA_PATH}")
print(f"Found classes: {subfolders}")

X_train = []
y_train = []

print("Extracting features for calibration (50 images per class)...")
for class_idx, class_name in enumerate(sorted(subfolders)):
    class_dir = os.path.join(DATA_PATH, class_name)
    images_list = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]
    
    print(f" -> Processing class '{class_name}' ({len(images_list)} images)...")
    for img_name in images_list:
        try:
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Ekstrakcja wektora cech przez załadowany model
            features = model(img_array, training=False).numpy()[0]
            X_train.append(features)
            y_train.append(class_idx)
        except Exception as e:
            print(f"    [WARN] Failed to process {img_name}: {e}")
            continue

if len(X_train) == 0:
    print("\n[ERROR] No calibration images were loaded! Check the warnings above.")
    exit()

print(f"\nTotal samples successfully loaded: {len(X_train)}. Training linear classifier head...")

# Trening mini-klasyfikatora
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

metadata = {
    'classifier': clf,
    'classes': sorted(subfolders)
}

HEAD_PATH = os.path.join('model', 'classifier_head.pkl')
with open(HEAD_PATH, 'wb') as f:
    pickle.dump(metadata, f)

print(f"[SUCCESS] Classifier head calibrated and saved to: {HEAD_PATH}")