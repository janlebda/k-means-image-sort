import os
import sys
import pickle

os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

print("==================================================")
print("     CALIBRATED CLASSIFIER - MODEL V1             ")
print("==================================================\n")

print("Drag and drop an image here and press Enter:")
image_path = input("> ").strip().replace("'", "").replace('"', '').replace("\\ ", " ")

MODEL_PATH = os.path.join('model', 'model_v1.keras')
HEAD_PATH = os.path.join('model', 'classifier_head.pkl')

if not os.path.exists(image_path):
    print("[ERROR] Image file does not exist!")
    input("Press Enter to exit...")
    sys.exit()

if not os.path.exists(HEAD_PATH):
    print("[ERROR] Missing calibration file! Run: python calibrate_classifier.py first.")
    input("Press Enter to exit...")
    sys.exit()

# 1. Load model and the calibrated classifier head
model = tf.keras.models.load_model(MODEL_PATH)
with open(HEAD_PATH, 'rb') as f:
    saved_data = pickle.load(f)

clf = saved_data['classifier']
class_labels = saved_data['classes']

# 2. Process image
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 3. Extract embedding vector
feature_vector = model(img_array, training=False).numpy()

# 4. Predict using the calibrated head
class_pred = clf.predict(feature_vector)[0]
probabilities = clf.predict_proba(feature_vector)[0]

detected_class = class_labels[class_pred].upper()
confidence = probabilities[class_pred] * 100

print("\n" + "="*50)
print("                PREDICTION RESULT                 ")
print("="*50)
print(f"  DETECTED CLASS:  ►► {detected_class} ◄◄")
print(f"  CONFIDENCE:      {confidence:.2f}%")
print("="*50)

print("\n" + "-"*50)
input("Press Enter to close the program...")