import os
import sys
import numpy as np
import pandas as pd
import joblib

from src.logger import logging
from src.exception import CustomException
from src.components.anomaly_detection import AnomalyDetection
from src.components.risk_engine import RiskEngine


class PredictPipeline:
    """
    Prediction pipeline for real-time transaction fraud detection.

    Loads:
        - preprocessor.pkl
        - feature_schema.pkl
        - best_model.pkl
        - isolation_forest.pkl

    Performs:
        1. Feature engineering
        2. Feature schema validation
        3. Preprocessing
        4. Supervised fraud prediction
        5. Isolation Forest anomaly detection
        6. Risk scoring
        7. APPROVE / REVIEW / BLOCK decision
    """

    def __init__(self):
        self.artifact_dir = "artifacts"

        self.preprocessor_path = os.path.join(
            self.artifact_dir,
            "preprocessor.pkl"
        )

        self.feature_schema_path = os.path.join(
            self.artifact_dir,
            "feature_schema.pkl"
        )

        self.model_path = os.path.join(
            self.artifact_dir,
            "models",
            "best_model.pkl"
        )

        self.isolation_forest_path = os.path.join(
            self.artifact_dir,
            "models",
            "isolation_forest.pkl"
        )

        self.preprocessor = None
        self.feature_schema = None
        self.model = None
        self.isolation_forest = None

    # ============================================================
    # LOAD ARTIFACTS
    # ============================================================

    def load_artifacts(self):
        """
        Load all trained artifacts.
        """

        try:
            logging.info("Loading prediction artifacts.")

            self.preprocessor = joblib.load(
                self.preprocessor_path
            )

            self.feature_schema = joblib.load(
                self.feature_schema_path
            )

            self.model = joblib.load(
                self.model_path
            )

            self.isolation_forest = joblib.load(
                self.isolation_forest_path
            )

            logging.info(
                "Prediction artifacts loaded successfully."
            )

        except Exception as e:
            logging.error(
                "Failed to load prediction artifacts.",
                exc_info=True
            )
            raise CustomException(e, sys)

    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================

    def create_features(self, df):
        """
        Create exactly the same features that were created
        during model training.

        IMPORTANT:
        TIMESTAMP is temporarily converted to datetime for
        calculating time differences, then converted back
        to numeric because the trained preprocessor expects
        TIMESTAMP as a numerical feature.
        """

        try:

            df = df.copy()

            logging.info("Creating prediction features.")

            # ----------------------------------------------------
            # Validate required columns
            # ----------------------------------------------------

            required_columns = [
                "TX_ID",
                "SENDER_ACCOUNT_ID",
                "RECEIVER_ACCOUNT_ID",
                "TX_TYPE",
                "TX_AMOUNT",
                "TIMESTAMP",
                "SENDER_CUSTOMER_ID",
                "RECEIVER_CUSTOMER_ID",
                "SENDER_COUNTRY",
                "RECEIVER_COUNTRY",
                "SENDER_ACCOUNT_TYPE",
                "RECEIVER_ACCOUNT_TYPE",
                "SENDER_INIT_BALANCE",
                "RECEIVER_INIT_BALANCE",
                "SENDER_TX_BEHAVIOR_ID",
                "RECEIVER_TX_BEHAVIOR_ID",
            ]

            missing_columns = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing input columns: {missing_columns}"
                )

            # ----------------------------------------------------
            # Keep original timestamp
            # ----------------------------------------------------

            original_timestamp = df["TIMESTAMP"].copy()

            # Convert temporarily to datetime.
            #
            # This is needed ONLY for calculating:
            # TIME_SINCE_SENDER_TX
            # TIME_SINCE_RECEIVER_TX
            #
            # We will convert TIMESTAMP back to numeric later.
            # ----------------------------------------------------

            try:
                timestamp_datetime = pd.to_datetime(
                    df["TIMESTAMP"],
                    errors="raise"
                )
            except Exception:

                # If timestamp is already numeric
                # convert it into datetime-like values
                #
                # This branch is mainly for compatibility.
                timestamp_datetime = pd.to_datetime(
                    df["TIMESTAMP"],
                    unit="s",
                    errors="coerce"
                )

            df["_TIMESTAMP_DATETIME"] = timestamp_datetime

            # ----------------------------------------------------
            # Sort transactions
            # ----------------------------------------------------

            df = df.sort_values(
                by=["_TIMESTAMP_DATETIME", "TX_ID"]
            ).reset_index(drop=True)

            # ====================================================
            # SENDER HISTORICAL FEATURES
            # ====================================================

            df["SENDER_TX_COUNT_BEFORE"] = (
                df.groupby("SENDER_ACCOUNT_ID")
                .cumcount()
            )

            df["_SENDER_CUM_AMOUNT"] = (
                df.groupby("SENDER_ACCOUNT_ID")["TX_AMOUNT"]
                .cumsum()
            )

            df["SENDER_AMOUNT_BEFORE"] = (
                df["_SENDER_CUM_AMOUNT"]
                - df["TX_AMOUNT"]
            )

            df["SENDER_AVG_AMOUNT_BEFORE"] = np.where(
                df["SENDER_TX_COUNT_BEFORE"] > 0,
                df["SENDER_AMOUNT_BEFORE"]
                / df["SENDER_TX_COUNT_BEFORE"],
                0.0
            )

            df["AMOUNT_TO_SENDER_AVG_RATIO"] = np.where(
                df["SENDER_AVG_AMOUNT_BEFORE"] > 0,
                df["TX_AMOUNT"]
                / df["SENDER_AVG_AMOUNT_BEFORE"],
                0.0
            )

            # ====================================================
            # RECEIVER HISTORICAL FEATURES
            # ====================================================

            df["RECEIVER_TX_COUNT_BEFORE"] = (
                df.groupby("RECEIVER_ACCOUNT_ID")
                .cumcount()
            )

            df["_RECEIVER_CUM_AMOUNT"] = (
                df.groupby("RECEIVER_ACCOUNT_ID")["TX_AMOUNT"]
                .cumsum()
            )

            df["RECEIVER_AMOUNT_BEFORE"] = (
                df["_RECEIVER_CUM_AMOUNT"]
                - df["TX_AMOUNT"]
            )

            df["RECEIVER_AVG_AMOUNT_BEFORE"] = np.where(
                df["RECEIVER_TX_COUNT_BEFORE"] > 0,
                df["RECEIVER_AMOUNT_BEFORE"]
                / df["RECEIVER_TX_COUNT_BEFORE"],
                0.0
            )

            # ====================================================
            # TIME SINCE PREVIOUS SENDER TRANSACTION
            # ====================================================

            df["TIME_SINCE_SENDER_TX"] = (
                df.groupby("SENDER_ACCOUNT_ID")
                ["_TIMESTAMP_DATETIME"]
                .diff()
                .dt.total_seconds()
            )

            # ====================================================
            # TIME SINCE PREVIOUS RECEIVER TRANSACTION
            # ====================================================

            df["TIME_SINCE_RECEIVER_TX"] = (
                df.groupby("RECEIVER_ACCOUNT_ID")
                ["_TIMESTAMP_DATETIME"]
                .diff()
                .dt.total_seconds()
            )

            # First transaction for an account
            # gets -1 exactly like training behavior.
            df["TIME_SINCE_SENDER_TX"] = (
                df["TIME_SINCE_SENDER_TX"]
                .fillna(-1)
            )

            df["TIME_SINCE_RECEIVER_TX"] = (
                df["TIME_SINCE_RECEIVER_TX"]
                .fillna(-1)
            )

            # ====================================================
            # NEW ACCOUNT FLAGS
            # ====================================================

            df["IS_NEW_SENDER"] = (
                df["SENDER_TX_COUNT_BEFORE"] == 0
            ).astype(int)

            df["IS_NEW_RECEIVER"] = (
                df["RECEIVER_TX_COUNT_BEFORE"] == 0
            ).astype(int)

            # ====================================================
            # IMPORTANT:
            # RESTORE TIMESTAMP TO NUMERIC
            # ====================================================

            #
            # Training used TIMESTAMP as a numerical feature.
            #
            # The training log confirms:
            #
            # Numerical columns included:
            #     'TIMESTAMP'
            #
            # Therefore the prediction DataFrame must also contain
            # TIMESTAMP as numeric before calling:
            #
            #     preprocessor.transform()
            #
            # ----------------------------------------------------

            if pd.api.types.is_numeric_dtype(original_timestamp):
                df["TIMESTAMP"] = pd.to_numeric(
                    original_timestamp,
                    errors="coerce"
                )

            else:

                # If user supplied a datetime string, convert it
                # into Unix timestamp seconds.
                #
                # This keeps TIMESTAMP numerical.
                parsed_timestamp = pd.to_datetime(
                    original_timestamp,
                    errors="coerce"
                )

                if parsed_timestamp.isna().any():
                    raise ValueError(
                        "Invalid TIMESTAMP value."
                    )

                df["TIMESTAMP"] = (
                    parsed_timestamp.astype("int64")
                    / 10**9
                )

            # ====================================================
            # REMOVE TEMPORARY COLUMNS
            # ====================================================

            df.drop(
                columns=[
                    "_TIMESTAMP_DATETIME",
                    "_SENDER_CUM_AMOUNT",
                    "_RECEIVER_CUM_AMOUNT",
                ],
                inplace=True,
                errors="ignore"
            )

            logging.info(
                "Prediction feature engineering completed."
            )

            return df

        except Exception as e:
            logging.error(
                "Feature engineering failed.",
                exc_info=True
            )
            raise CustomException(e, sys)

    # ============================================================
    # REMOVE UNWANTED COLUMNS
    # ============================================================

    def remove_unwanted_columns(self, df):
        """
        Remove IDs, leakage columns and columns that were not
        used by the training model.
        """

        columns_to_drop = [
            "TX_ID",
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "SENDER_CUSTOMER_ID",
            "RECEIVER_CUSTOMER_ID",
            "ALERT_ID",
            "TX_TYPE",
            "SENDER_COUNTRY",
            "RECEIVER_COUNTRY",
            "SENDER_ACCOUNT_TYPE",
            "RECEIVER_ACCOUNT_TYPE",
            "SENDER_AMOUNT_BEFORE",
            "RECEIVER_AMOUNT_BEFORE",
            "IS_FRAUD",
        ]

        existing_columns = [
            col
            for col in columns_to_drop
            if col in df.columns
        ]

        df = df.drop(
            columns=existing_columns,
            errors="ignore"
        )

        logging.info(
            f"Dropped prediction columns: {existing_columns}"
        )

        return df

    # ============================================================
    # PREPARE INPUT
    # ============================================================

    def prepare_input(self, transaction):
        """
        Convert transaction dictionary/DataFrame into the exact
        feature structure expected by the trained model.
        """

        try:

            # ----------------------------------------------------
            # Convert input to DataFrame
            # ----------------------------------------------------

            if isinstance(transaction, dict):

                df = pd.DataFrame([transaction])

            elif isinstance(transaction, pd.DataFrame):

                df = transaction.copy()

            else:

                raise TypeError(
                    "transaction must be a dictionary "
                    "or pandas DataFrame."
                )

            # ----------------------------------------------------
            # Feature engineering
            # ----------------------------------------------------

            df = self.create_features(df)

            # ----------------------------------------------------
            # Remove columns not used by model
            # ----------------------------------------------------

            df = self.remove_unwanted_columns(df)

            # ----------------------------------------------------
            # Validate feature schema
            # ----------------------------------------------------

            # feature_schema.pkl contains metadata dictionary
# Example:
# {
#     "feature_columns": [...],
#     "numerical_columns": [...],
#     "categorical_columns": [...],
#     "constant_columns": [...],
#     "target_column": "IS_FRAUD"
# }

            if isinstance(self.feature_schema, dict):

                 expected_features = self.feature_schema[
        "feature_columns"
                 ]

            else:

    # Backward compatibility if schema is directly
    # stored as a list
                expected_features = list(
             self.feature_schema
             )

            actual_features = list(
                df.columns
            )

            missing_features = [
                feature
                for feature in expected_features
                if feature not in actual_features
            ]

            extra_features = [
                feature
                for feature in actual_features
                if feature not in expected_features
            ]

            if missing_features:

                raise ValueError(
                    f"Missing prediction features: "
                    f"{missing_features}"
                )

            if extra_features:

                logging.warning(
                    f"Extra prediction features will be "
                    f"removed: {extra_features}"
                )

                df = df.drop(
                    columns=extra_features
                )

            # ----------------------------------------------------
            # EXACT SAME ORDER AS TRAINING
            # ----------------------------------------------------

            df = df[
                expected_features
            ]

            # ----------------------------------------------------
            # Final numeric conversion
            # ----------------------------------------------------

            if "TIMESTAMP" in df.columns:

                df["TIMESTAMP"] = pd.to_numeric(
                    df["TIMESTAMP"],
                    errors="coerce"
                )

            # ----------------------------------------------------
            # Check numeric columns for invalid values
            # ----------------------------------------------------

            df.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True
            )

            logging.info(
                f"Prediction input shape: {df.shape}"
            )

            logging.info(
                f"Prediction features: "
                f"{list(df.columns)}"
            )

            return df

        except Exception as e:

            logging.error(
                "Failed to prepare prediction input.",
                exc_info=True
            )

            raise CustomException(e, sys)

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(self, transaction):
        """
        Run the complete fraud prediction pipeline."""
    

        try:
             logging.info("Starting transaction prediction.")

        # 1. Load trained artifacts
             self.load_artifacts()

        # 2. Prepare transaction features
             X = self.prepare_input(transaction)

        # 3. Apply the fitted preprocessor
             logging.info("Applying trained preprocessor.")

             X_transformed = self.preprocessor.transform(X)

             logging.info(
            f"Transformed prediction shape: {X_transformed.shape}"
         )

        # 4. Predict fraud probability using the supervised model
             fraud_probability = float(
            self.model.predict_proba(X_transformed)[0, 1]
         )

             fraud_prediction = int(fraud_probability >= 0.5)

        # 5. Detect anomalies using the new component
             anomaly_detector = AnomalyDetection()

             anomaly_result = anomaly_detector.detect(X_transformed)[0]

             isolation_prediction = anomaly_result[
            "isolation_prediction"
        ]

             is_anomaly = anomaly_result["is_anomaly"]

             isolation_decision = anomaly_result[
            "isolation_decision_score"
        ]

             anomaly_score = 1.0 if is_anomaly else 0.0

        # 6. Calculate the final risk score and decision
             risk_engine = RiskEngine()

             risk_result = risk_engine.calculate_risk(
            fraud_probability=fraud_probability,
            is_anomaly=is_anomaly
        )

        # 7. Prepare the response
             result = {
            "fraud_probability": round(
                fraud_probability, 6
            ),
            "fraud_prediction": fraud_prediction,
            "isolation_prediction": isolation_prediction,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 6),
            "isolation_decision_score": round(
                isolation_decision, 6
            ),
            "risk_score": risk_result["risk_score"],
            "decision": risk_result["decision"]
        }

             logging.info(
            f"Fraud probability: {result['fraud_probability']}"
        )

             logging.info(
            f"Anomaly detected: {result['is_anomaly']}"
        )

             logging.info(
            f"Risk score: {result['risk_score']}"
        )

             logging.info(
            f"Final decision: {result['decision']}"
        )

             logging.info("Transaction prediction completed.")

             return result

        except Exception as e:
              logging.error(
            "Prediction failed.",
            exc_info=True
        )

              raise CustomException(e, sys)


# ================================================================
# LOCAL TEST
# ================================================================

if __name__ == "__main__":

    try:

        # --------------------------------------------------------
        # Create predictor
        # --------------------------------------------------------

        predictor = PredictPipeline()

        # --------------------------------------------------------
        # Sample transaction
        #
        # IMPORTANT:
        # TIMESTAMP is numeric because your training data used
        # TIMESTAMP as a numerical feature.
        # --------------------------------------------------------

        sample_transaction = {

            "TX_ID": "TEST_TX_001",

            "SENDER_ACCOUNT_ID": "ACC_0001",

            "RECEIVER_ACCOUNT_ID": "ACC_0002",

            "TX_TYPE": "TRANSFER",

            "TX_AMOUNT": 5000.0,

            "TIMESTAMP": 200,

            "SENDER_CUSTOMER_ID": "CUST_0001",

            "RECEIVER_CUSTOMER_ID": "CUST_0002",

            "SENDER_COUNTRY": "IND",

            "RECEIVER_COUNTRY": "IND",

            "SENDER_ACCOUNT_TYPE": "SAVINGS",

            "RECEIVER_ACCOUNT_TYPE": "SAVINGS",

            "SENDER_INIT_BALANCE": 50000.0,

            "RECEIVER_INIT_BALANCE": 30000.0,

            "SENDER_TX_BEHAVIOR_ID": 1,

            "RECEIVER_TX_BEHAVIOR_ID": 1,

        }

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        result = predictor.predict(
            sample_transaction
        )

        # --------------------------------------------------------
        # Print result
        # --------------------------------------------------------

        print()
        print("=" * 60)
        print("TRANSACTION FRAUD PREDICTION")
        print("=" * 60)

        for key, value in result.items():

            print(
                f"{key:30}: {value}"
            )

        print("=" * 60)

    except Exception as e:

        logging.error(
            "Prediction pipeline failed.",
            exc_info=True
        )

        raise CustomException(e, sys)