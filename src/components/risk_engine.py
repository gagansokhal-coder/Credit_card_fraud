import sys

from src.logger import logging
from src.exception import CustomException


class RiskEngine:

    def __init__(
        self,
        fraud_weight=0.75,
        anomaly_weight=0.25,
        review_threshold=0.50,
        block_threshold=0.80
    ):
        try:
            if fraud_weight < 0 or anomaly_weight < 0:
                raise ValueError("Weights cannot be negative.")

            if fraud_weight + anomaly_weight <= 0:
                raise ValueError("At least one weight must be positive.")

            if not 0 <= review_threshold <= 1:
                raise ValueError("Review threshold must be between 0 and 1.")

            if not 0 <= block_threshold <= 1:
                raise ValueError("Block threshold must be between 0 and 1.")

            if review_threshold > block_threshold:
                raise ValueError(
                    "Review threshold cannot exceed block threshold."
                )

            total_weight = fraud_weight + anomaly_weight

            self.fraud_weight = fraud_weight / total_weight
            self.anomaly_weight = anomaly_weight / total_weight

            self.review_threshold = review_threshold
            self.block_threshold = block_threshold

        except Exception as e:
            raise CustomException(e, sys)

    def calculate_risk(self, fraud_probability, is_anomaly):
        try:
            fraud_probability = float(fraud_probability)

            if not 0 <= fraud_probability <= 1:
                raise ValueError(
                    "Fraud probability must be between 0 and 1."
                )

            anomaly_score = 1.0 if is_anomaly else 0.0

            risk_score = (
                self.fraud_weight * fraud_probability
                + self.anomaly_weight * anomaly_score
            )

            risk_score = max(0.0, min(1.0, risk_score))

            if risk_score >= self.block_threshold:
                decision = "BLOCK"

            elif risk_score >= self.review_threshold:
                decision = "REVIEW"

            else:
                decision = "APPROVE"

            result = {
                "fraud_probability": fraud_probability,
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": anomaly_score,
                "risk_score": round(risk_score, 4),
                "decision": decision
            }

            logging.info(
                "Risk assessment completed: %s",
                decision
            )

            return result

        except Exception as e:
            raise CustomException(e, sys)