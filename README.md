# Image Clusterer — K-Means + MobileNetV2

Grupuje obrazy w podfolderach na podstawie ich wizualnej zawartości,
używając cech wyodrębnionych przez pretrenowaną sieć MobileNetV2 oraz
algorytmu K-Means.

## Struktura projektu

```
image_clusterer/
├── cluster_images.py   ← główny skrypt
├── requirements.txt
├── dataset/            ← tu wrzuć obrazy (np. wypakuj dataset z Kaggle)
│   ├── PetImages/
│   │   ├── Cat/
│   │   └── Dog/
├── clustered/          ← tu trafią posortowane pliki (tworzone automatycznie)
│   ├── cluster_0/
│   └── cluster_1/
└── logs/               ← logi każdego uruchomienia (tworzone automatycznie)
```

## Szybki start

### 1. Pobierz dataset
Pobierz Microsoft Cats vs Dogs z Kaggle:
https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

Wypakuj do folderu `dataset/` wewnątrz projektu.

### 2. Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### 3. Uruchom
```bash
# Domyślnie: k=2 klastry, pliki z folderu dataset/, wynik w clustered/
python cluster_images.py

# Własne parametry:
python cluster_images.py --images-dir dataset --output-dir clustered --k 2

# Przesuń pliki zamiast kopiować (oszczędność miejsca):
python cluster_images.py --move
```

## Wszystkie opcje

| Argument | Domyślnie | Opis |
|---|---|---|
| `--images-dir` / `-i` | `dataset` | Folder z obrazami (względem skryptu) |
| `--output-dir` / `-o` | `clustered` | Folder wyjściowy na klastry |
| `--k` / `-k` | `2` | Liczba klastrów |
| `--batch-size` / `-b` | `32` | Obrazy na batch (zmniejsz przy błędach pamięci) |
| `--pca-variance` / `-p` | `0.95` | Zachowana wariancja po PCA (0–1) |
| `--move` | wyłączone | Przesuń pliki zamiast kopiować |

## Jak to działa

```
Obrazy → MobileNetV2 → wektor 1280 cech → PCA (redukcja) → K-Means → cluster_N/
```

1. **Skanowanie** — skrypt rekurencyjnie szuka obrazów w `--images-dir`
2. **Ekstrakcja cech** — MobileNetV2 (ImageNet) jako ekstraktor cech wizualnych
3. **PCA** — redukcja wymiarów z zachowaniem 95% wariancji
4. **K-Means** — grupowanie z inicjalizacją k-means++
5. **Kopiowanie** — pliki trafiają do `clustered/cluster_0/`, `clustered/cluster_1/` itd.

> **Wskazówka:** Dla datasetu Cats vs Dogs użyj `--k 2`.
> Sprawdź logi — podają rozmiar każdego klastra, dzięki czemu widać,
> czy klaster 0 to koty a klaster 1 to psy (lub odwrotnie).
