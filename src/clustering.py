import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def reduce_dimensions(features: np.ndarray, variance: float, logger: logging.Logger) -> np.ndarray:
    logger.info("=" * 60)
    logger.info("STEP 3 — Dimensionality reduction with PCA")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    reduced = pca.fit_transform(scaled)
    logger.info(f"  Reduced dimensions: {reduced.shape[1]}")
    return reduced

def run_clustering(features: np.ndarray, k: int, logger: logging.Logger) -> np.ndarray:
    logger.info("=" * 60)
    logger.info("STEP 4 — K-Means clustering")
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)
    
    for c in range(k):
        count = int((labels == c).sum())
        logger.info(f"    Cluster {c:>2}: {count:>5} images")
    return labels

def visualize_clusters(features: np.ndarray, labels: np.ndarray, k: int, output_path: str):
    """Visualize clusters in 2D using PCA."""
    pca_2d = PCA(n_components=2)
    vis_data = pca_2d.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(vis_data[:, 0], vis_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(f"K-Means Clustering Visualization (k={k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")
