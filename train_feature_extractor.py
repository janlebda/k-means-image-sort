import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.cnn_model import create_feature_extractor, create_training_model

def train_model(data_dir, epochs=10, batch_size=32, model_save_path='model/feature_extractor.h5'):
    """
    Trains a CNN on the images in the train folder to learn feature extraction.
    """
    # 1. Data Loading
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # Use 20% of train for internal validation
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    # 2. Model Creation
    extractor = create_feature_extractor(embedding_dim=256)
    full_model = create_training_model(extractor, num_classes=train_generator.num_classes)

    # 3. Training
    print("Starting training...")
    full_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # 4. Save the extractor part
    # We only care about the part that produces the data vectors
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    extractor.save(model_save_path)
    print(f"Feature extractor saved to {model_save_path}")

if __name__ == "__main__":
    TRAIN_PATH = os.path.join('data', 'train')
    train_model(TRAIN_PATH)
