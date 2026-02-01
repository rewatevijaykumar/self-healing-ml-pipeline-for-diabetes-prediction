import logging
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# =========================
# Logging utilities
# =========================

def setup_logging(log_dir="logs", log_file="pipeline.log"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def log_event(message, level="info"):
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    else:
        logging.info(message)

# =========================
# Data statistics utilities
# =========================

def compute_numeric_stats(df, numeric_cols):
    """
    Compute summary statistics for numeric columns
    Used for validation and drift detection
    """
    stats = df[numeric_cols].agg(["min", "max", "mean", "std"]).to_dict()
    return stats

# =========================
# Drift detection utilities
# =========================

def detect_drift(df_new, numeric_stats, threshold=2.0):
    """
    Detect data drift based on mean shift
    """
    drifted_columns = []

    for col, stats in numeric_stats.items():
        mean_train = stats["mean"]
        std_train = stats["std"]
        mean_new = df_new[col].mean()

        if abs(mean_new - mean_train) > threshold * std_train:
            drifted_columns.append(col)

    return drifted_columns

# =========================
# Model versioning utilities
# =========================

def generate_versioned_filename(base_name, extension="pkl"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def save_metadata(metadata, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    filename = generate_versioned_filename("metadata", "json")
    path = os.path.join(output_dir, filename)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

    return path
