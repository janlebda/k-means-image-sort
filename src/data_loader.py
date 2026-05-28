import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMG_SIZE = (224, 224)

def collect_image_paths(root: Path, logger: logging.Logger) -> list[Path]:
    """
    Recursively scan *root* for image files.
    Skips any subfolder that looks like a cluster output (starts with 'cluster_').
    """
    logger.info("=" * 60)
    logger.info("STEP 1 — Scanning for images")
    logger.info(f"  Root directory : {root}")

    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip previously created cluster folders
        dirnames[:] = [d for d in dirnames if not d.lower().startswith("cluster_")]

        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTS:
                paths.append(Path(dirpath) / fname)

    logger.info(f"  Found {len(paths)} image(s)")
    if len(paths) == 0:
        logger.error("  No images found!")
        return []

    return paths

def load_and_preprocess(path: Path) -> np.ndarray | None:
    """
    Open one image, resize to 224×224, convert to RGB, apply MobileNetV2 preprocessing.
    """
    try:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        return preprocess_input(arr)
    except Exception:
        return None
