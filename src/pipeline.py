import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os, json, datetime, logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def build_pipeline(num_cols, cat_cols):
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    transformers = [('num', num_transformer, num_cols)]
    if cat_cols:
        transformers.append(('cat', cat_transformer, cat_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', Ridge(alpha=1.0))])
    return pipeline

def save_model(pipeline, numeric_stats, mse, r2):
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/pipeline_{timestamp}.pkl"
    joblib.dump(pipeline, model_path)
    
    metadata = {
        "timestamp": timestamp,
        "numeric_stats": numeric_stats,
        "mse": mse,
        "r2": r2
    }
    with open(f"models/pipeline_{timestamp}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info(f"Saved model {model_path} with metrics MSE={mse:.4f}, R2={r2:.4f}")
    return model_path
