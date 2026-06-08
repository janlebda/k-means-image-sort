import tensorflow as tf
from tensorflow.keras import layers, models
import os

def create_feature_extractor(input_shape=(160, 160, 3), embedding_dim=512):
    """
    Creates a CNN model that outputs a feature vector (embedding) for a given image.
    This can be trained as part of an autoencoder or a classification task.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers to create the feature vector
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
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
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example initialization
    extractor = create_feature_extractor()
    training_wrapper = create_training_model(extractor)
    
    extractor.summary()
    print("\nCNN Model created successfully.")
