import os
import sys
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException


# =========================================================
# Configuration
# =========================================================

@dataclass
class DataTransformationConfig:

    preprocessor_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )

    train_data_path: str = os.path.join(
        "artifacts",
        "train_data.pkl"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test_data.pkl"
    )

    feature_schema_path: str = os.path.join(
        "artifacts",
        "feature_schema.pkl"
    )


# =========================================================
# Data Transformation
# =========================================================

class DataTransformation:

    def __init__(self):

        self.transformation_config = (
            DataTransformationConfig()
        )

    # =====================================================
    # Historical Feature Engineering
    # =====================================================

    def create_features(self, df):

        try:

            logging.info(
                "Starting feature engineering."
            )

            df = df.copy()

            # -------------------------------------------------
            # Sort chronologically
            # -------------------------------------------------

            df = df.sort_values(
                by=["TIMESTAMP", "TX_ID"]
            ).reset_index(drop=True)

            # -------------------------------------------------
            # Sender historical transaction count
            #
            # cumcount() counts only PREVIOUS transactions.
            # Therefore it does not use future information.
            # -------------------------------------------------

            df["SENDER_TX_COUNT_BEFORE"] = (
                df.groupby("SENDER_ACCOUNT_ID")
                .cumcount()
            )

            # -------------------------------------------------
            # Sender historical amount
            # -------------------------------------------------

            sender_amount_sum = (
                df.groupby("SENDER_ACCOUNT_ID")["TX_AMOUNT"]
                .cumsum()
            )

            df["SENDER_AMOUNT_BEFORE"] = (
                sender_amount_sum - df["TX_AMOUNT"]
            )

            # -------------------------------------------------
            # Sender historical average amount
            # -------------------------------------------------

            df["SENDER_AVG_AMOUNT_BEFORE"] = np.where(
                df["SENDER_TX_COUNT_BEFORE"] > 0,

                df["SENDER_AMOUNT_BEFORE"]
                / df["SENDER_TX_COUNT_BEFORE"],

                np.nan
            )

            # -------------------------------------------------
            # Amount ratio compared with sender history
            # -------------------------------------------------

            df["AMOUNT_TO_SENDER_AVG_RATIO"] = np.where(
                df["SENDER_AVG_AMOUNT_BEFORE"] > 0,

                df["TX_AMOUNT"]
                / df["SENDER_AVG_AMOUNT_BEFORE"],

                np.nan
            )

            # -------------------------------------------------
            # Receiver historical transaction count
            # -------------------------------------------------

            df["RECEIVER_TX_COUNT_BEFORE"] = (
                df.groupby("RECEIVER_ACCOUNT_ID")
                .cumcount()
            )

            # -------------------------------------------------
            # Receiver historical amount
            # -------------------------------------------------

            receiver_amount_sum = (
                df.groupby("RECEIVER_ACCOUNT_ID")["TX_AMOUNT"]
                .cumsum()
            )

            df["RECEIVER_AMOUNT_BEFORE"] = (
                receiver_amount_sum - df["TX_AMOUNT"]
            )

            # -------------------------------------------------
            # Receiver historical average
            # -------------------------------------------------

            df["RECEIVER_AVG_AMOUNT_BEFORE"] = np.where(
                df["RECEIVER_TX_COUNT_BEFORE"] > 0,

                df["RECEIVER_AMOUNT_BEFORE"]
                / df["RECEIVER_TX_COUNT_BEFORE"],

                np.nan
            )

            # -------------------------------------------------
            # Time since previous sender transaction
            # -------------------------------------------------

            df["TIME_SINCE_SENDER_TX"] = (
                df.groupby("SENDER_ACCOUNT_ID")["TIMESTAMP"]
                .diff()
            )

            # -------------------------------------------------
            # Time since previous receiver transaction
            # -------------------------------------------------

            df["TIME_SINCE_RECEIVER_TX"] = (
                df.groupby("RECEIVER_ACCOUNT_ID")["TIMESTAMP"]
                .diff()
            )

            # -------------------------------------------------
            # New sender indicator
            # -------------------------------------------------

            df["IS_NEW_SENDER"] = (
                df["SENDER_TX_COUNT_BEFORE"] == 0
            ).astype(int)

            # -------------------------------------------------
            # New receiver indicator
            # -------------------------------------------------

            df["IS_NEW_RECEIVER"] = (
                df["RECEIVER_TX_COUNT_BEFORE"] == 0
            ).astype(int)

            logging.info(
                "Historical feature engineering completed."
            )

            return df

        except Exception as e:

            logging.exception(
                "Error during feature engineering."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Time-Based Train/Test Split
    # =====================================================

    def split_data(self, df):

        try:

            logging.info(
                "Starting time-based train/test split."
            )

            # Make sure data is sorted
            df = df.sort_values(
                by=["TIMESTAMP", "TX_ID"]
            ).reset_index(drop=True)

            unique_timestamps = sorted(
                df["TIMESTAMP"].unique()
            )

            # 80% of time points for training
            split_index = int(
                len(unique_timestamps) * 0.80
            )

            train_timestamps = (
                unique_timestamps[:split_index]
            )

            test_timestamps = (
                unique_timestamps[split_index:]
            )

            train_data = df[
                df["TIMESTAMP"].isin(
                    train_timestamps
                )
            ].copy()

            test_data = df[
                df["TIMESTAMP"].isin(
                    test_timestamps
                )
            ].copy()

            logging.info(
                f"Training timestamps: "
                f"{train_timestamps[0]} "
                f"to "
                f"{train_timestamps[-1]}"
            )

            logging.info(
                f"Testing timestamps: "
                f"{test_timestamps[0]} "
                f"to "
                f"{test_timestamps[-1]}"
            )

            logging.info(
                f"Train shape: {train_data.shape}"
            )

            logging.info(
                f"Test shape: {test_data.shape}"
            )

            return train_data, test_data

        except Exception as e:

            logging.exception(
                "Error during train/test split."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Remove unwanted / leakage columns
    # =====================================================

    def remove_unwanted_columns(self, df):

        try:

            df = df.copy()

            columns_to_drop = [

                # ---------------------------------------------
                # Identifiers
                # ---------------------------------------------

                "TX_ID",

                "SENDER_ACCOUNT_ID",

                "RECEIVER_ACCOUNT_ID",

                "SENDER_CUSTOMER_ID",

                "RECEIVER_CUSTOMER_ID",

                "ALERT_ID",

                # ---------------------------------------------
                # Original transaction type
                #
                # Your current dataset has only TRANSFER,
                # so it contains no useful variation.
                # ---------------------------------------------

                "TX_TYPE",

                # ---------------------------------------------
                # Constant account information in your dataset
                #
                # COUNTRY = US
                # ACCOUNT_TYPE = I
                #
                # These provide no variation.
                # ---------------------------------------------

                "SENDER_COUNTRY",

                "RECEIVER_COUNTRY",

                "SENDER_ACCOUNT_TYPE",

                "RECEIVER_ACCOUNT_TYPE",

                # ---------------------------------------------
                # Intermediate feature
                # ---------------------------------------------

                "SENDER_AMOUNT_BEFORE",

                "RECEIVER_AMOUNT_BEFORE"
            ]

            # Remove only columns that actually exist
            columns_to_drop = [
                col
                for col in columns_to_drop
                if col in df.columns
            ]

            df = df.drop(
                columns=columns_to_drop
            )

            logging.info(
                f"Dropped columns: {columns_to_drop}"
            )

            return df

        except Exception as e:

            logging.exception(
                "Error while removing columns."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Create preprocessing object
    # =====================================================

    def get_preprocessor(
        self,
        X_train
    ):

        try:

            # -------------------------------------------------
            # Identify numerical columns
            # -------------------------------------------------

            numerical_columns = (
                X_train
                .select_dtypes(
                    include=["int64", "float64", "int32", "float32"]
                )
                .columns
                .tolist()
            )

            # -------------------------------------------------
            # Identify categorical columns
            # -------------------------------------------------

            categorical_columns = (
                X_train
                .select_dtypes(
                    include=["object", "category", "string"]
                )
                .columns
                .tolist()
            )

            logging.info(
                f"Numerical columns: "
                f"{numerical_columns}"
            )

            logging.info(
                f"Categorical columns: "
                f"{categorical_columns}"
            )

            # -------------------------------------------------
            # Numerical pipeline
            # -------------------------------------------------

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # -------------------------------------------------
            # Categorical pipeline
            # -------------------------------------------------

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True
                        )
                    )
                ]
            )

            # -------------------------------------------------
            # Column Transformer
            # -------------------------------------------------

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numerical_pipeline,
                        numerical_columns
                    ),

                    (
                        "cat",
                        categorical_pipeline,
                        categorical_columns
                    )
                ],

                remainder="drop"
            )

            return (
                preprocessor,
                numerical_columns,
                categorical_columns
            )

        except Exception as e:

            logging.exception(
                "Error while creating preprocessor."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Main Transformation Method
    # =====================================================

    def initiate_data_transformation(
        self,
        merged_data
    ):

        try:

            logging.info(
                "========== DATA TRANSFORMATION STARTED =========="
            )

            # -------------------------------------------------
            # Step 1
            # Feature engineering
            # -------------------------------------------------

            df = self.create_features(
                merged_data
            )

            # -------------------------------------------------
            # Step 2
            # Time-based split
            # -------------------------------------------------

            train_data, test_data = (
                self.split_data(df)
            )

            # -------------------------------------------------
            # Step 3
            # Separate target
            # -------------------------------------------------

            target_column = "IS_FRAUD"

            y_train = (
                train_data[target_column]
                .astype(int)
                .values
            )

            y_test = (
                test_data[target_column]
                .astype(int)
                .values
            )

            # -------------------------------------------------
            # Step 4
            # Remove target
            # -------------------------------------------------

            X_train = train_data.drop(
                columns=[target_column]
            )

            X_test = test_data.drop(
                columns=[target_column]
            )

            # -------------------------------------------------
            # Step 5
            # Remove unwanted columns
            # -------------------------------------------------

            X_train = self.remove_unwanted_columns(
                X_train
            )

            X_test = self.remove_unwanted_columns(
                X_test
            )

            # -------------------------------------------------
            # Step 6
            # Find constant columns using TRAIN only
            # -------------------------------------------------

            constant_columns = [
                col
                for col in X_train.columns
                if X_train[col].nunique(
                    dropna=False
                ) <= 1
            ]

            if constant_columns:

                logging.info(
                    f"Removing constant columns: "
                    f"{constant_columns}"
                )

                X_train = X_train.drop(
                    columns=constant_columns
                )

                X_test = X_test.drop(
                    columns=constant_columns,
                    errors="ignore"
                )

            # -------------------------------------------------
            # Step 7
            # Create preprocessor
            # -------------------------------------------------

            (
                preprocessor,
                numerical_columns,
                categorical_columns
            ) = self.get_preprocessor(
                X_train
            )

            # -------------------------------------------------
            # Step 8
            # Fit ONLY on training data
            # -------------------------------------------------

            logging.info(
                "Fitting preprocessor on training data."
            )

            X_train_transformed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            # -------------------------------------------------
            # Step 9
            # Transform test data
            # -------------------------------------------------

            logging.info(
                "Transforming test data."
            )

            X_test_transformed = (
                preprocessor.transform(
                    X_test
                )
            )

            # -------------------------------------------------
            # Step 10
            # Save preprocessor
            # -------------------------------------------------

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            joblib.dump(
                preprocessor,
                self.transformation_config.preprocessor_path
            )

            # -------------------------------------------------
            # Step 11
            # Save feature schema
            # -------------------------------------------------

            feature_schema = {

                "feature_columns": X_train.columns.tolist(),

                "numerical_columns": numerical_columns,

                "categorical_columns": categorical_columns,

                "constant_columns": constant_columns,

                "target_column": target_column
            }

            joblib.dump(
                feature_schema,
                self.transformation_config.feature_schema_path
            )

            # -------------------------------------------------
            # Step 12
            # Save train/test data
            # -------------------------------------------------

            train_data_transformed = (
                X_train_transformed,
                y_train
            )

            test_data_transformed = (
                X_test_transformed,
                y_test
            )

            joblib.dump(
                train_data_transformed,
                self.transformation_config.train_data_path
            )

            joblib.dump(
                test_data_transformed,
                self.transformation_config.test_data_path
            )

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            logging.info(
                f"Transformed train shape: "
                f"{X_train_transformed.shape}"
            )

            logging.info(
                f"Transformed test shape: "
                f"{X_test_transformed.shape}"
            )

            logging.info(
                f"Fraud samples in train: "
                f"{y_train.sum()}"
            )

            logging.info(
                f"Fraud samples in test: "
                f"{y_test.sum()}"
            )

            logging.info(
                "========== DATA TRANSFORMATION COMPLETED =========="
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test
            )

        except Exception as e:

            logging.exception(
                "Data transformation failed."
            )

            raise CustomException(e, sys)


# =========================================================
# Local Testing
# =========================================================

if __name__ == "__main__":

    try:

        from src.components.data_ingestion import (
            DataIngestion
        )

        logging.info(
            "Testing DataTransformation locally."
        )

        # ---------------------------------------------
        # Run ingestion
        # ---------------------------------------------

        ingestion = DataIngestion()

        (
            merged_data,
            alerts_data,
            merged_path
        ) = ingestion.initiate_data_ingestion()

        # ---------------------------------------------
        # Run transformation
        # ---------------------------------------------

        transformation = DataTransformation()

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = transformation.initiate_data_transformation(
            merged_data
        )

        print(
            "\nTraining feature shape:",
            X_train.shape
        )

        print(
            "Testing feature shape:",
            X_test.shape
        )

        print(
            "Training target shape:",
            y_train.shape
        )

        print(
            "Testing target shape:",
            y_test.shape
        )

        print(
            "\nTraining fraud count:",
            y_train.sum()
        )

        print(
            "Testing fraud count:",
            y_test.sum()
        )

    except Exception as e:

        logging.exception(
            "Data transformation failed during local testing."
        )

        raise CustomException(e, sys)