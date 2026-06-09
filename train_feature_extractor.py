import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# POPRAWIONO: Import z pliku cnn.py w folderze model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model.cnn_model import create_feature_extractor, create_training_model

def train_model(train_dir, test_dir, epochs=20, batch_size=16, model_save_path='model/feature_extractor.keras'):
    """
    Trenuje sieć CNN, generuje statystyki, zapisuje raport i sam ekstraktor cech.
    """
    t_start = time.time()
    
    # 1. Przygotowanie danych (Trening + Walidacja w locie)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% zbioru train idzie na walidację
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    # Ostateczny test (czyste zdjęcia, bez modyfikacji/augmentacji)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False  # Przy teście nie mieszamy kolejności
    )

    # 2. Tworzenie modeli
    extractor = create_feature_extractor(input_shape=(128, 128, 3), embedding_dim=512)
    full_model = create_training_model(extractor, num_classes=train_generator.num_classes)


    stop_early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    # 3. Trening
    print("\n=== ROZPOCZĘCIE TRENINGU ===")
    history = full_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[stop_early, reduce_lr]
    )

    # 4. Ewaluacja końcowa na zbiorze TESTOWYM (Egzamin)
    print("\n=== URUCHAMIANIE EGZAMINU KOŃCOWEGO (ZBIÓR TESTOWY) ===")
    test_loss, test_acc = full_model.evaluate(test_generator)
    print(f"Dokładność na zbiorze testowym: {test_acc * 100:.2f}%")

    # 5. Zapisywanie samego ekstraktora (wagi + architektura) do przyszłego klastrowanie
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    extractor.save(model_save_path)
    print(f"\n[SUKCES] Ekstraktor cech zapisany do: {model_save_path}")

    # 6. GENEROWANIE PEŁNEGO RAPORTU STATYSTYCZNEGO DO PLIKU
    t_end = time.time()
    duration_minutes = (t_end - t_start) / 60
    
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"logs/raport_treningu_{timestamp}.log"
    
    # Wyciągamy najlepsze wyniki z historii uczenia
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("          RAPORT STATYSTYCZNY MODELU CNN          \n")
        f.write("==================================================\n")
        f.write(f"Data utworzenia:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Czas trwania treningu: {duration_minutes:.2f} minut\n")
        f.write(f"Liczba epok:           {epochs}\n")
        f.write(f"Rozmiar paczki (batch):{batch_size}\n")
        f.write("--------------------------------------------------\n")
        f.write("DANE METRYCZNE ZBIORU:\n")
        f.write(f"  - Zdjęcia treningowe: {train_generator.samples}\n")
        f.write(f"  - Zdjęcia walidacyjne:{val_generator.samples}\n")
        f.write(f"  - Zdjęcia testowe:    {test_generator.samples}\n")
        f.write(f"  - Liczba klas:         {train_generator.num_classes}\n")
        f.write("--------------------------------------------------\n")
        f.write("WYNIKI KLASYFIKACJI (DOKŁADNOŚĆ / ACCURACY):\n")
        f.write(f"  [Najlepsza] Dokładność treningowa:   {best_train_acc * 100:.2f}%\n")
        f.write(f"  [Najlepsza] Dokładność walidacyjna:  {best_val_acc * 100:.2f}%\n")
        f.write(f"  ==> OSTATECZNA DOKŁADNOŚĆ TESTOWA:   {test_acc * 100:.2f}%\n")
        f.write("--------------------------------------------------\n")
        f.write("ARCHITEKTURA EKSTRAKTORA:\n")
        # Zapisuje strukturę sieci do pliku tekstowego
        extractor.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n==================================================\n")

    print(f"[SUKCES] Raport statystyczny zapisany do: {log_file_path}")

if __name__ == "__main__":
    # Ustawiamy ścieżki względne do folderu głównego projektu
    TRAIN_PATH = os.path.join('data', 'train')
    TEST_PATH = os.path.join('data', 'test')
    
    # Wywołanie funkcji z obiema ścieżkami (możesz zmienić liczbę epok np. na 15 lub 20)
    train_model(TRAIN_PATH, TEST_PATH, epochs=8, batch_size=32)