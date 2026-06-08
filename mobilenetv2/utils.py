import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(log_dir: Path) -> logging.Logger:
    """Configure a logger that writes to both console and a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = "[%(asctime)s]  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("ImageClusterer")
    logger.info(f"Log file: {log_file}")
    return logger
