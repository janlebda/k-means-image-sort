import os
import sys
import argparse
import time
import shutil
import numpy as np
from pathlib import Path
from utils import setup_logger
from data_loader import collect_image_paths
from feature_extractor import FeatureExtractor
from clustering import reduce_dimensions, run_clustering, visualize_clusters
from evaluator import calculate_accuracy

def copy_to_clusters(paths: list[Path], labels: np.ndarray, output_root: Path, logger, move=False):
    logger.info("=" * 60)
    logger.info(f"STEP 6 — {'Moving' if move else 'Copying'} images to clusters")
    k = int(labels.max()) + 1
    for c in range(k):
        (output_root / f"cluster_{c}").mkdir(parents=True, exist_ok=True)
    
    for idx, (src, label) in enumerate(zip(paths, labels)):
        dst = output_root / f"cluster_{label}" / src.name
        if dst.exists():
            dst = output_root / f"cluster_{label}" / f"{src.stem}_{idx}{src.suffix}"
        try:
            if move: shutil.move(str(src), dst)
            else: shutil.copy2(src, dst)
        except Exception as e:
            logger.warning(f"  Failed to transfer {src.name}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image Clusterer")
    parser.add_argument("--images-dir", "-i", help="Folder containing images")
    parser.add_argument("--output-dir", "-o", default="clustered", help="Output folder")
    parser.add_argument("--k", "-k", type=int, default=2, help="Number of clusters")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--pca-variance", "-p", type=float, default=0.95, help="PCA variance")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Folder selection logic
    images_dir = args.images_dir
    if not images_dir:
        print("Available directories:")
        current_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
        for i, d in enumerate(current_dirs):
            print(f"[{i}] {d}")
        
        try:
            choice = input("Select images directory (enter number or path): ")
            if choice.isdigit() and int(choice) < len(current_dirs):
                images_dir = current_dirs[int(choice)]
            else:
                images_dir = choice
        except EOFError:
            images_dir = "dataset" # Default if no input
            
    images_dir = Path(images_dir).resolve()
    if not images_dir.exists():
        print(f"Error: Directory {images_dir} does not exist.")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / args.output_dir
    logs_dir = project_root / "logs"
    logger = setup_logger(logs_dir)

    logger.info(f"Using images from: {images_dir}")

    # Pipeline
    all_paths = collect_image_paths(images_dir, logger)
    if not all_paths: return

    extractor = FeatureExtractor(logger)
    features, valid_paths = extractor.extract(all_paths, args.batch_size)

    reduced = reduce_dimensions(features, args.pca_variance, logger)
    labels = run_clustering(reduced, args.k, logger)
    
    # Visualization
    vis_path = logs_dir / f"clusters_{time.strftime('%Y%m%d_%H%M%S')}.png"
    visualize_clusters(reduced, labels, args.k, str(vis_path))

    # Accuracy
    calculate_accuracy(valid_paths, labels, logger)

    # Transfer
    copy_to_clusters(valid_paths, labels, output_dir, logger, move=args.move)
    
    logger.info("=" * 60)
    logger.info("Done!")

if __name__ == "__main__":
    main()
