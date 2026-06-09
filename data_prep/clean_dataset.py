import os
from PIL import Image

def clean_folder(target_dir):
    """
    Przeszukuje folder i usuwa pliki, których biblioteka PIL nie potrafi otworzyć
    (czyli pliki uszkodzone lub o rozmiarze 0 bajtów).
    """
    removed_count = 0
    print(f"Skanowanie folderu: {target_dir}...")
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # 1. Sprawdzenie czy plik nie jest pusty
                if os.path.getsize(file_path) == 0:
                    print(f"[USUWANIE] Pusty plik: {file_path}")
                    os.remove(file_path)
                    removed_count += 1
                    continue
                
                # 2. Próba otwarcia pliku przez PIL
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Weryfikacja struktury pliku
                except Exception:
                    print(f"[USUWANIE] Uszkodzony plik: {file_path}")
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"Nie udało się usunąć {file_path}: {e}")
                        
    print(f"Zakończono czyszczenie {target_dir}. Usunięto uszkodzonych plików: {removed_count}\n")

if __name__ == "__main__":
    # Czyścimy zarówno zbiór treningowy, jak i testowy
    TRAIN_PATH = os.path.join('data', 'train')
    TEST_PATH = os.path.join('data', 'test')
    
    if not os.path.exists(TRAIN_PATH):
        print("Folder 'data/train' nie istnieje. Uruchom najpierw split_dataset.py!")
    else:
        clean_folder(TRAIN_PATH)
        clean_folder(TEST_PATH)
        print("Baza danych jest teraz czysta i gotowa do bezpiecznego treningu.")