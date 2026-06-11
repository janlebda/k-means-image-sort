import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Ukrycie zbędnych logów TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def visualize_clusters(features: np.ndarray, labels: np.ndarray, k: int, output_path: str):
    """Generuje i zapisuje wykres klastrów 2D za pomocą zredukowanych cech."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Klastry")
    plt.title(f"Wizualizacja K-Means po Normalizacji L2 (K={k})")
    plt.xlabel("Główna składowa PCA 1")
    plt.ylabel("Główna składowa PCA 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Wykres zapisany do: {output_path}")

def extract_features_and_cluster(input_folder, output_folder, model_path='model/feature_extractor.keras', num_clusters=2, target_size=(128, 128)):
    print(f"=== Wczytywanie modelu z {model_path} ===")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")
    
    # Wczytanie autorskiego ekstraktora
    extractor = tf.keras.models.load_model(model_path, compile=False)
    image_paths, images_data = [], []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    print(f"\n=== Skanowanie folderów w {input_folder} ===")
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(root, filename)
                try:
                    # Ładowanie i reskalowanie (128x128, 1./255 tak jak w Twoim treningu)
                    img = image.load_img(img_path, target_size=target_size)
                    img_array = image.img_to_array(img)
                    img_array = img_array / 255.0 
                    images_data.append(img_array)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Błąd podczas wczytywania {img_path}: {e}")

    if not images_data:
        print("Nie znaleziono obrazów.")
        return

    images_batch = np.array(images_data)
    print(f"Przetworzono {len(image_paths)} zdjęć. Rozpoczęcie ekstrakcji cech...")
    
    # Pobranie surowych wektorów (Liczba_zdjęć x 512)
    raw_features = extractor.predict(images_batch, batch_size=16)
    print(f"  Oryginalny kształt cech: {raw_features.shape}")

    print("\n=== Redukcja i Normalizacja Danych ===")
    # 1. KLUCZOWY KROK DLA TWOJEJ SIECI CNN: Normalizacja L2
    # Przekształca wektory tak, by odległość euklidesowa oddawała podobieństwo kątowe
    l2_features = normalize(raw_features, norm='l2')
    print("  Wykonano normalizację L2.")
    
    # 2. PCA: Usuwamy szum i spłaszczamy wymiary (zostawiamy 90% informacji)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=42)
    reduced_features = pca.fit_transform(l2_features)
    print(f"  Kształt po PCA: {reduced_features.shape}")

    print(f"\n=== Klastrowanie K-Means (K={num_clusters}) ===")
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    for c in range(num_clusters):
        count = int((cluster_labels == c).sum())
        print(f"  Klaster {c}: {count} zdjęć")

    print("\n=== Kopiowanie plików oraz Generowanie Wizualizacji ===")
    for i in range(num_clusters):
        os.makedirs(os.path.join(output_folder, f'klaster_{i}'), exist_ok=True)

    for idx, (img_path, label) in enumerate(zip(image_paths, cluster_labels)):
        safe_filename = f"{idx}_{os.path.basename(img_path)}"
        destination = os.path.join(output_folder, f'klaster_{label}', safe_filename)
        shutil.copy(img_path, destination)

    # Generowanie wykresu PCA 2D do podglądu (używamy tylko 2 pierwszych komponentów do rysowania)
    plot_path = os.path.join(output_folder, 'wizualizacja_klastrow.png')
    pca_2d = PCA(n_components=2)
    vis_data = pca_2d.fit_transform(l2_features)
    visualize_clusters(vis_data, cluster_labels, num_clusters, plot_path)
        
    print(f"\n[SUKCES] Posortowane obrazy i wykres znajdują się w: {output_folder}")

if __name__ == "__main__":
    # Konfiguracja ścieżek
    INPUT_DIR = 'data/test'
    OUTPUT_DIR = 'posortowane'
    MODEL_PATH = 'model/feature_extractor.keras'

    os.makedirs(INPUT_DIR, exist_ok=True)
    has_files = any(files for _, _, files in os.walk(INPUT_DIR))
    
    if not has_files:
        print(f"Wrzuć zdjęcia do folderu '{INPUT_DIR}' i uruchom skrypt ponownie.")
    else:
        extract_features_and_cluster(
            input_folder=INPUT_DIR,
            output_folder=OUTPUT_DIR,
            model_path=MODEL_PATH,
            num_clusters=2
        )