import tensorflow as tf
from tensorflow.keras import layers, models
import os

def create_feature_extractor(input_shape=(128, 128, 3), embedding_dim=512):
    """
    Creates a CNN model that outputs a feature vector (embedding) for a given image.
    This can be trained as part of an autoencoder or a classification task.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Blok 1: Rozbity na osobne kroki (brak ukrytej fuzji cuDNN)
        layers.Conv2D(32, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(), # <-- Teraz aktywacja jest osobną, bezpieczną warstwą!
        layers.MaxPooling2D((2, 2)),
        
        # Blok 2
        layers.Conv2D(64, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 3
        layers.Conv2D(128, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 4
        layers.Conv2D(256, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Spłaszczenie i gęste warstwy
        layers.Flatten(),
        layers.Dense(512, activation=None),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        layers.Dense(embedding_dim, activation=None, name='feature_vector')
    ])
    
    return model

def create_training_model(feature_extractor, num_classes=2):
    """
    Wraps the feature extractor in a classification head for training.
    """
    model = models.Sequential([
        feature_extractor,
        layers.ReLU(), # Add activation before classification
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=False)
    return model

if __name__ == "__main__":
    # Example initialization
    extractor = create_feature_extractor()
    training_wrapper = create_training_model(extractor)
    
    extractor.summary()
    print("\nCNN Model created successfully.")
