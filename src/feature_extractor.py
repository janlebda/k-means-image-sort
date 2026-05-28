import time
import logging
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from data_loader import load_and_preprocess

class FeatureExtractor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model = self._build_model()

    def _build_model(self) -> Model:
        self.logger.info("=" * 60)
        self.logger.info("STEP 2 — Building feature extractor (MobileNetV2)")
        base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                           input_shape=(224, 224, 3))
        base.trainable = False
        model = Model(inputs=base.input, outputs=base.output)
        self.logger.info("  Model loaded  : MobileNetV2 (ImageNet weights)")
        return model

    def extract(self, paths: list[Path], batch_size: int) -> tuple[np.ndarray, list[Path]]:
        self.logger.info("=" * 60)
        self.logger.info("STEP 2 — Extracting features from images")
        
        features = []
        valid_paths = []
        failed = 0
        total = len(paths)
        t0 = time.time()

        for batch_start in range(0, total, batch_size):
            batch_paths = paths[batch_start: batch_start + batch_size]
            batch_imgs = []
            batch_valid = []

            for p in batch_paths:
                arr = load_and_preprocess(p)
                if arr is None:
                    failed += 1
                else:
                    batch_imgs.append(arr)
                    batch_valid.append(p)

            if batch_imgs:
                batch_tensor = np.stack(batch_imgs, axis=0)
                batch_feats = self.model.predict(batch_tensor, verbose=0)
                features.append(batch_feats)
                valid_paths.extend(batch_valid)

            done = min(batch_start + batch_size, total)
            elapsed = time.time() - t0
            pct = done / total * 100
            self.logger.info(f"  Progress: {done:>5}/{total}  ({pct:5.1f}%)")

        feature_matrix = np.vstack(features)
        self.logger.info(f"  Extraction complete. Successful: {len(valid_paths)}, Failed: {failed}")
        return feature_matrix, valid_paths
