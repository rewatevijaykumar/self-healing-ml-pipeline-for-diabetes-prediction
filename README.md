# self-healing-ml-pipeline-for-diabetes-prediction
Self-Healing ML Pipeline for Diabetes Prediction

Project Overview

This project implements a self-healing ML pipeline for predicting diabetes progression using the sklearn diabetes dataset. The pipeline handles:

Feature preprocessing (numeric scaling, categorical encoding)

Anomaly detection and data cleaning

Data drift monitoring

Automatic retraining with versioned model storage

Logging and metrics tracking

Key Features

Production-ready pipeline

Versioned models & metadata

Logging & alerting for anomalies and drift

Easy extension to other datasets

flowchart TD
    A[Data Ingestion] --> B[Preprocessing]
    B --> C[Feature Scaling & Encoding]
    C --> D[Anomaly Detection]
    D --> E{Anomaly or Drift Detected?}
    E -- No --> F[Prediction]
    E -- Yes --> G[Retraining Pipeline]
    G --> F
    F --> H[Logging & Model Versioning]
    H --> I[Deployed Model / Monitoring]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#f96,stroke:#333,stroke-width:2px
    style E fill:#faa,stroke:#333,stroke-width:2px
    style G fill:#6f6,stroke:#333,stroke-width:2px
    style F fill:#cff,stroke:#333,stroke-width:2px
    style H fill:#fcf,stroke:#333,stroke-width:2px
    style I fill:#afa,stroke:#333,stroke-width:2px

Usage Example

from src.predictor import SelfHealingPredictor
import pandas as pd

# Load new data
X_new = pd.read_csv("data/diabetes.csv")

# Initialize predictor
predictor = SelfHealingPredictor(model_path="models/pipeline_20260101_120000.pkl",
                                 numeric_stats={...})

# Predict and self-heal if needed
predictions = predictor.predict(X_new, target_col="target")
print(predictions[:5])

self_healing_ml_pipeline/
│
├─ data/
│   └─ diabetes.csv          # optional: dataset copy
│
├─ notebooks/
│   └─ exploratory_analysis.ipynb
│
├─ src/
│   ├─ pipeline.py           # main pipeline code
│   ├─ predictor.py          # self-healing predictor class
│   └─ utils.py              # helper functions (logging, metrics)
│
├─ models/
│   └─ (versioned models go here)
│
├─ logs/
│   └─ pipeline.log
│
├─ README.md
└─ requirements.txt
