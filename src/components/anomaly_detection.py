import os
import sys
import joblib
import numpy as np

from pathlib import Path

from src.logger import logging
from src.exception import CustomException


class AnomalyDetection:

    def __init__(self, model_path=None):
        try:
            if model_path is None:
                project_root = Path(__file__).resolve().parents[2]
                model_path = (
                    project_root
                    / "artifacts"
                    / "models"
                    / "isolation_forest.pkl"
                )

            self.model_path = Path(model_path)
            self.model = self.load_model()

        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Isolation Forest model not found: {self.model_path}"
                )

            model = joblib.load(self.model_path)

            logging.info("Isolation Forest model loaded successfully.")

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def detect(self, X):
        try:
            if X is None or len(X) == 0:
                raise ValueError(
                    "Input data cannot be empty."
                )

            predictions = self.model.predict(X)

            decision_scores = self.model.decision_function(X)

            results = []

            for prediction, score in zip(
                predictions,
                decision_scores
            ):
                is_anomaly = int(prediction) == -1

                results.append({
                    "isolation_prediction": int(prediction),
                    "is_anomaly": bool(is_anomaly),
                    "isolation_decision_score": float(score)
                })

            logging.info(
                "Anomaly detection completed successfully."
            )

            return results

        except Exception as e:
            raise CustomException(e, sys)