import logging
import numpy as np
from pathlib import Path
from scipy.stats import mode

def calculate_accuracy(paths: list[Path], labels: np.ndarray, logger: logging.Logger):
    """
    Calculate clustering accuracy by mapping each cluster to the most frequent ground truth label.
    Assumes parent directory name is the ground truth label.
    """
    logger.info("=" * 60)
    logger.info("STEP 5 — Calculating Accuracy")
    
    # Extract ground truth labels from path parents
    # e.g., dataset/PetImages/Cat/1.jpg -> "Cat"
    true_labels_str = [p.parent.name for p in paths]
    unique_true_labels = sorted(list(set(true_labels_str)))
    label_to_id = {label: i for i, label in enumerate(unique_true_labels)}
    true_labels = np.array([label_to_id[l] for l in true_labels_str])
    
    k = len(np.unique(labels))
    match_count = 0
    
    cluster_to_true = {}
    
    for i in range(k):
        # Find the most frequent true label in this cluster
        mask = (labels == i)
        if not np.any(mask):
            continue
            
        relevant_true_labels = true_labels[mask]
        most_common = mode(relevant_true_labels, keepdims=True).mode[0]
        
        cluster_to_true[i] = unique_true_labels[most_common]
        match_count += np.sum(relevant_true_labels == most_common)
        
        logger.info(f"  Cluster {i}: assigned to label '{unique_true_labels[most_common]}'")

    accuracy = match_count / len(labels)
    logger.info(f"  Overall Clustering Accuracy: {accuracy * 100:.2f}%")
    return accuracy
