import pandas as pd
import numpy as np
import joblib
import logging, time
from sklearn.metrics import mean_squared_error, r2_score
from pipeline import save_model
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

class SelfHealingPredictor:
    def __init__(self, model_path, numeric_stats, anomaly_threshold=0.1, drift_threshold=2.0, cooldown_sec=60):
        self.pipeline = joblib.load(model_path)
        self.numeric_stats = numeric_stats
        self.anomaly_threshold = anomaly_threshold
        self.drift_threshold = drift_threshold
        self.cooldown_sec = cooldown_sec
        self.last_retrain_time = 0
        self.model_path = model_path

        self.num_cols = self.pipeline.named_steps['preprocessor'].transformers_[0][2]
        self.cat_cols = []
        if len(self.pipeline.named_steps['preprocessor'].transformers_) > 1:
            self.cat_cols = self.pipeline.named_steps['preprocessor'].transformers_[1][2]

    def validate_and_monitor(self, df_new):
        df_checked = df_new.copy()
        anomalies = 0

        # Numeric validation
        for col in self.num_cols:
            df_checked[col] = df_checked[col].fillna(df_checked[col].mean())
            col_min = self.numeric_stats[col]['min']
            col_max = self.numeric_stats[col]['max']
            df_checked[col] = df_checked[col].clip(lower=col_min, upper=col_max)
            if (df_checked[col] < col_min).any() or (df_checked[col] > col_max).any():
                anomalies += 1

        fraction_anomalies = anomalies / len(self.num_cols)
        anomaly_detected = fraction_anomalies > self.anomaly_threshold

        # Drift detection
        drift_cols = []
        for col in self.num_cols:
            mean_train = self.numeric_stats[col]['mean']
            std_train = self.numeric_stats[col]['std']
            mean_new = df_checked[col].mean()
            if abs(mean_new - mean_train) > self.drift_threshold * std_train:
                drift_cols.append(col)
        drift_detected = len(drift_cols) > 0

        return df_checked, anomaly_detected or drift_detected, drift_cols

    def retrain(self, df_new, target_col='target'):
        X_new = df_new.drop(columns=[target_col])
        y_new = df_new[target_col]

        pipeline_new = Pipeline([
            ('preprocessor', self.pipeline.named_steps['preprocessor']),
            ('model', Ridge(alpha=1.0))
        ])
        pipeline_new.fit(X_new, y_new)
        y_pred = pipeline_new.predict(X_new)
        mse = mean_squared_error(y_new, y_pred)
        r2 = r2_score(y_new, y_pred)
        self.pipeline = pipeline_new
        self.model_path = save_model(pipeline_new, self.numeric_stats, mse, r2)
        self.last_retrain_time = time.time()
        logging.info(f"Retrained model with new MSE={mse:.4f}, R2={r2:.4f}")

    def predict(self, df_new, target_col=None):
        df_checked, retrain_needed, drift_cols = self.validate_and_monitor(df_new)
        if retrain_needed and target_col and target_col in df_checked.columns:
            if time.time() - self.last_retrain_time > self.cooldown_sec:
                self.retrain(df_checked, target_col)
        df_model = df_checked[self.num_cols + self.cat_cols]
        preds = self.pipeline.predict(df_model)
        logging.info(f"Predicted {len(preds)} samples, drifted columns: {drift_cols}")
        return preds
