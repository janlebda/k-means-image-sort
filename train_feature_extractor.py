import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Wyciszenie logów TF (INFO i WARNING)

import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model.cnn_model import create_feature_extractor, create_training_model

# ============================================================
# ZMIANY W TYM PLIKU:
#   1. Usunięta augmentacja z ImageDataGenerator (przeniesiona do modelu)
#   2. Dodany callback ModelCheckpoint — zapisuje najlepszy model w trakcie
#   3. Domyślny batch_size zmniejszony do 32 (lepiej dla 4GB VRAM)
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Włącza dynamiczne przydzielanie VRAM (tylko tyle, ile potrzeba)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Pomyślnie włączono dynamiczną alokację pamięci VRAM.")
    except RuntimeError as e:
        print("Błąd konfiguracji GPU:", e)

def train_model(
    train_dir,
    test_dir,
    epochs=20,
    batch_size=32,  # ZMIANA: było 128, teraz 32 — bezpieczniejsze dla 4GB VRAM
    model_save_path='model/feature_extractor.keras'
):
    """
    Trenuje sieć CNN, generuje statystyki, zapisuje raport i ekstraktor cech.

    Argumenty:
        train_dir       -- folder ze zdjęciami treningowymi (podkatalogi = klasy)
        test_dir        -- folder ze zdjęciami testowymi
        epochs          -- maksymalna liczba epok (EarlyStopping może przerwać wcześniej)
        batch_size      -- ile zdjęć na raz trafia do GPU
        model_save_path -- gdzie zapisać wytrenowany ekstraktor
    """
    t_start = time.time()

    # -------------------------------------------------------
    # 1. PRZYGOTOWANIE DANYCH
    # -------------------------------------------------------
    # ZMIANA: usunięta augmentacja z generatora (rotation, flip, shift)
    # bo augmentacja jest teraz wbudowana w model jako warstwy RandomFlip itp.
    # Generator robi już tylko rescale (normalizację 0-255 → 0.0-1.0)
    # i podział na train/validation.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 20% zdjęć treningowych → walidacja
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

    # Zbiór testowy — bez augmentacji, bez mieszania kolejności
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    # -------------------------------------------------------
    # 2. TWORZENIE MODELU
    # -------------------------------------------------------
    extractor = create_feature_extractor(input_shape=(128, 128, 3), embedding_dim=512)
    full_model = create_training_model(extractor, num_classes=train_generator.num_classes)

    # -------------------------------------------------------
    # 3. CALLBACKI
    # -------------------------------------------------------
    # EarlyStopping: przerywa trening gdy val_loss przestaje spadać
    # patience=4 oznacza: czekaj 4 epoki bez poprawy, potem stop
    # restore_best_weights: przywraca wagi z najlepszej epoki
    stop_early = EarlyStopping(
        monitor='val_loss',
        patience=4,           # było 3, teraz 4 — dajemy sieci więcej szansy
        restore_best_weights=True,
        verbose=1
    )

    # ReduceLROnPlateau: zmniejsza learning rate gdy trening staje w miejscu
    # factor=0.5 → learning rate ×0.5 (np. 0.001 → 0.0005)
    # patience=2 → po 2 epokach bez poprawy
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,  # ZMIANA: dodany dolny limit żeby LR nie spadł do zera
        verbose=1
    )

    # NOWY callback: ModelCheckpoint
    # Zapisuje model po każdej epoce TYLKO jeśli val_accuracy się poprawiła.
    # Dzięki temu nie stracisz najlepszego modelu jeśli trening się posypie.
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint_path = model_save_path.replace('.keras', '_checkpoint.keras')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,  # zapisuje tylko gdy jest poprawa
        verbose=1
    )

    # -------------------------------------------------------
    # 4. TRENING
    # -------------------------------------------------------
    print("\n=== ROZPOCZĘCIE TRENINGU ===")
    print(f"GPU: {[d.name for d in tf.config.list_physical_devices('GPU')]}")
    history = full_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[stop_early, reduce_lr, checkpoint]  # ZMIANA: dodany checkpoint
    )

    # -------------------------------------------------------
    # 5. EWALUACJA NA ZBIORZE TESTOWYM
    # -------------------------------------------------------
    print("\n=== EGZAMIN KOŃCOWY (ZBIÓR TESTOWY) ===")
    test_loss, test_acc = full_model.evaluate(test_generator)
    print(f"Dokładność na zbiorze testowym: {test_acc * 100:.2f}%")

    # -------------------------------------------------------
    # 6. ZAPIS EKSTRAKTORA (sam ekstraktor, bez głowicy klasyfikacyjnej)
    # -------------------------------------------------------
    extractor.save(model_save_path)
    print(f"\n[SUKCES] Ekstraktor cech zapisany do: {model_save_path}")

    # -------------------------------------------------------
    # 7. RAPORT STATYSTYCZNY
    # -------------------------------------------------------
    t_end = time.time()
    duration_minutes = (t_end - t_start) / 60

    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"logs/raport_treningu_{timestamp}.log"

    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    actual_epochs = len(history.history['accuracy'])  # ile epok faktycznie przeszło

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("          RAPORT STATYSTYCZNY MODELU CNN          \n")
        f.write("==================================================\n")
        f.write(f"Data utworzenia:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Czas trwania treningu: {duration_minutes:.2f} minut\n")
        f.write(f"Liczba epok (max):     {epochs}\n")
        f.write(f"Liczba epok (real):    {actual_epochs}\n")  # NOWE: ile faktycznie
        f.write(f"Rozmiar paczki:        {batch_size}\n")
        f.write("--------------------------------------------------\n")
        f.write("DANE METRYCZNE ZBIORU:\n")
        f.write(f"  - Zdjęcia treningowe: {train_generator.samples}\n")
        f.write(f"  - Zdjęcia walidacyjne:{val_generator.samples}\n")
        f.write(f"  - Zdjęcia testowe:    {test_generator.samples}\n")
        f.write(f"  - Liczba klas:        {train_generator.num_classes}\n")
        f.write("--------------------------------------------------\n")
        f.write("WYNIKI KLASYFIKACJI:\n")
        f.write(f"  [Najlepsza] Dokładność treningowa:   {best_train_acc * 100:.2f}%\n")
        f.write(f"  [Najlepsza] Dokładność walidacyjna:  {best_val_acc * 100:.2f}%\n")
        f.write(f"  ==> OSTATECZNA DOKŁADNOŚĆ TESTOWA:   {test_acc * 100:.2f}%\n")
        f.write("--------------------------------------------------\n")
        f.write("ARCHITEKTURA EKSTRAKTORA:\n")
        extractor.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n==================================================\n")

    print(f"[SUKCES] Raport zapisany do: {log_file_path}")


if __name__ == "__main__":
    TRAIN_PATH = os.path.join('data', 'train')
    TEST_PATH = os.path.join('data', 'test')

    train_model(
        TRAIN_PATH,
        TEST_PATH,
        epochs=20,       # więcej epok, EarlyStopping przerwie jak trzeba
        batch_size=32    # bezpieczne dla GTX 1050 Ti 4GB
    )