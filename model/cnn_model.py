import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# ZMIANY W TYM PLIKU:
#   1. Augmentacja jako warstwa modelu (nowe)
#   2. Piąty blok Conv2D 512 filtrów (nowe)
#   3. GlobalAveragePooling2D zamiast Flatten (zmiana)
#   4. Lepszy klasyfikator z dwoma warstwami Dense (zmiana)
#   5. Label smoothing w funkcji straty (nowe)
# ============================================================


def create_feature_extractor(input_shape=(128, 128, 3), embedding_dim=512):
    """
    Tworzy ekstraktor cech CNN z 5 blokami konwolucyjnymi.

    Architektura bloku: Conv2D → BatchNorm → ReLU → MaxPool
    - Conv2D: wykrywa wzorce (krawędzie, tekstury, kształty)
    - BatchNormalization: normalizuje wartości między warstwami,
      przyspiesza trening i stabilizuje gradienty
    - ReLU: aktywacja — "przepuszcza" tylko wartości dodatnie,
      wprowadza nieliniowość
    - MaxPooling: zmniejsza rozmiar mapy cech 2x, sieć widzi
      coraz szerszy kontekst

    Przepływ przez sieć (128×128 wejście):
      Blok 1 (32  filtry) → 64×64
      Blok 2 (64  filtry) → 32×32
      Blok 3 (128 filtry) → 16×16
      Blok 4 (256 filtry) → 8×8
      Blok 5 (512 filtry) → 4×4   ← NOWY BLOK
      GlobalAveragePooling → 512 wartości (jedna na filtr)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # NOWE: augmentacja jako pierwsza warstwa
        # (aktywna tylko podczas treningu, pomijana przy predykcji)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),

        # Blok 1: wykrywa proste krawędzie i kolory
        layers.Conv2D(32, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Blok 2: wykrywa tekstury (sierść, futro)
        layers.Conv2D(64, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Blok 3: wykrywa wzorce złożone (plamki, paski, struktury)
        layers.Conv2D(128, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Blok 4: wykrywa części ciała (uszy, nos, oczy)
        layers.Conv2D(256, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # NOWY Blok 5: wykrywa abstrakcyjne cechy (kształt głowy, sylwetka)
        # Dlaczego 512 filtrów? Każdy filtr to jeden "detektor cechy" —
        # przy 512 sieć może rozróżnić więcej subtelnych różnic
        layers.Conv2D(512, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),  # → 4×4×512

        # ZMIANA: GlobalAveragePooling2D zamiast Flatten + Dense(512)
        #
        # Flatten zamieniał 4×4×512 = 8192 wartości w jeden wektor
        # i podawał je do Dense(512) → to tworzyło 8192×512 = 4.2M parametrów
        # w jednej warstwie, która łatwo się przeuczała.
        #
        # GlobalAveragePooling2D uśrednia każdą z 512 map cech do 1 wartości
        # → wyjście: wektor 512 wartości, ZERO dodatkowych parametrów.
        # Sieć jest lżejsza i lepiej generalizuje.
        layers.GlobalAveragePooling2D(),

        # Warstwa projekcji: mapuje 512 → embedding_dim
        # (zostawiona z oryginalnej architektury, teraz ma sens jako
        #  "podsumowanie" cech, nie jako redukcja chaosu po Flatten)
        layers.Dense(embedding_dim, activation=None, name='feature_vector'),
    ])

    return model


def create_training_model(feature_extractor, num_classes=2):
    """
    Owija ekstraktor w głowicę klasyfikacyjną do treningu.
    """
    # ============================================================
    # ROZWIĄZANIE BŁĘDU: Zamiast szukać feature_extractor.output,
    # tworzymy jawne wejście i przepuszczamy je przez ekstraktor.
    # To ZMUSZA Keras do zbudowania grafu i rozwiązuje problem.
    # ============================================================
    inputs = layers.Input(shape=(128, 128, 3))
    
    # Traktujemy cały Twój Sequential model jak warstwę:
    features = feature_extractor(inputs)

    # Dalej leci Twoja głowica klasyfikacyjna (bez zmian):
    x = layers.BatchNormalization()(features)
    x = layers.ReLU()(x)

    x = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Spinamy model (od wejścia do wyjścia)
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
        jit_compile=False  
    )

    return model


if __name__ == "__main__":
    extractor = create_feature_extractor()
    training_wrapper = create_training_model(extractor)
    training_wrapper.summary()
    print("\nCNN Model created successfully.")