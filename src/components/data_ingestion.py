import os
import pandas as pd
import sys

from dataclasses import dataclass
from pathlib import Path

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """

    project_root = Path(__file__).resolve().parents[2]

    accounts_path: str = str(
        project_root / "notebook" / "data" / "raw" / "accounts.csv"
    )

    transactions_path: str = str(
        project_root / "notebook" / "data" / "raw" / "transaction_data.csv"
    )

    alerts_path: str = str(
        project_root / "notebook" / "data" / "raw" / "alerts.csv"
    )

    # Output is only an intermediate artifact.
    merged_data_path: str = str(
        project_root / "artifacts" / "merged_transaction_data.csv"
    )


class DataIngestion:
  

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # ---------------------------------------------------------
    # Load raw datasets
    # ---------------------------------------------------------

    def load_data(self):
        try:
            logging.info("Starting raw data loading.")

            accounts = pd.read_csv(
                self.ingestion_config.accounts_path
            )

            transactions = pd.read_csv(
                self.ingestion_config.transactions_path
            )

            alerts = pd.read_csv(
                self.ingestion_config.alerts_path
            )

            logging.info(
                f"Accounts shape: {accounts.shape}"
            )

            logging.info(
                f"Transactions shape: {transactions.shape}"
            )

            logging.info(
                f"Alerts shape: {alerts.shape}"
            )

            return accounts, transactions, alerts

        except Exception as e:
            logging.exception("Error while loading raw datasets.")
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    def validate_columns(
        self,
        accounts,
        transactions,
        alerts
    ):
        try:

            required_accounts = {
                "ACCOUNT_ID",
                "CUSTOMER_ID",
                "INIT_BALANCE",
                "COUNTRY",
                "ACCOUNT_TYPE",
                "IS_FRAUD",
                "TX_BEHAVIOR_ID"
            }

            required_transactions = {
                "TX_ID",
                "SENDER_ACCOUNT_ID",
                "RECEIVER_ACCOUNT_ID",
                "TX_TYPE",
                "TX_AMOUNT",
                "TIMESTAMP",
                "IS_FRAUD",
                "ALERT_ID"
            }

            required_alerts = {
                "ALERT_ID",
                "ALERT_TYPE",
                "IS_FRAUD",
                "TX_ID",
                "SENDER_ACCOUNT_ID",
                "RECEIVER_ACCOUNT_ID",
                "TX_TYPE",
                "TX_AMOUNT",
                "TIMESTAMP"
            }

            missing_accounts = (
                required_accounts -
                set(accounts.columns)
            )

            missing_transactions = (
                required_transactions -
                set(transactions.columns)
            )

            missing_alerts = (
                required_alerts -
                set(alerts.columns)
            )

            if missing_accounts:
                raise ValueError(
                    f"Missing columns in accounts.csv: "
                    f"{missing_accounts}"
                )

            if missing_transactions:
                raise ValueError(
                    f"Missing columns in transaction_data.csv: "
                    f"{missing_transactions}"
                )

            if missing_alerts:
                raise ValueError(
                    f"Missing columns in alerts.csv: "
                    f"{missing_alerts}"
                )

            logging.info("Column validation successful.")

        except Exception as e:
            logging.exception("Column validation failed.")
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # Basic data validation
    # ---------------------------------------------------------

    def validate_data(
        self,
        accounts,
        transactions,
        alerts
    ):
        try:

            # -------------------------------
            # Null check
            # -------------------------------

            accounts_nulls = accounts.isnull().sum().sum()
            transactions_nulls = transactions.isnull().sum().sum()
            alerts_nulls = alerts.isnull().sum().sum()

            logging.info(
                f"Accounts null values: {accounts_nulls}"
            )

            logging.info(
                f"Transactions null values: {transactions_nulls}"
            )

            logging.info(
                f"Alerts null values: {alerts_nulls}"
            )

            # -------------------------------
            # Duplicate check
            # -------------------------------

            logging.info(
                f"Account duplicates: "
                f"{accounts.duplicated().sum()}"
            )

            logging.info(
                f"Transaction duplicates: "
                f"{transactions.duplicated().sum()}"
            )

            logging.info(
                f"Alert duplicates: "
                f"{alerts.duplicated().sum()}"
            )

            # -------------------------------
            # Primary key validation
            # -------------------------------

            if accounts["ACCOUNT_ID"].duplicated().any():
                raise ValueError(
                    "ACCOUNT_ID is not unique in accounts.csv."
                )

            if transactions["TX_ID"].duplicated().any():
                raise ValueError(
                    "TX_ID is not unique in transaction_data.csv."
                )

            # if alerts["ALERT_ID"].duplicated().any():
            #     raise ValueError(
            #         "ALERT_ID is not unique in alerts.csv."
            #     )

            logging.info("Basic data validation successful.")

        except Exception as e:
            logging.exception("Data validation failed.")
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # Merge account information
    # ---------------------------------------------------------

    def merge_account_data(
        self,
        accounts,
        transactions
    ):

        try:

            logging.info(
                "Starting sender account merge."
            )

            # -------------------------------------------------
            # Prepare SENDER account information
            # -------------------------------------------------

            sender_accounts = accounts[
                [
                    "ACCOUNT_ID",
                    "CUSTOMER_ID",
                    "INIT_BALANCE",
                    "COUNTRY",
                    "ACCOUNT_TYPE",
                    "TX_BEHAVIOR_ID"
                ]
            ].copy()

            sender_accounts = sender_accounts.rename(
                columns={
                    "ACCOUNT_ID": "SENDER_ACCOUNT_ID",
                    "CUSTOMER_ID": "SENDER_CUSTOMER_ID",
                    "INIT_BALANCE": "SENDER_INIT_BALANCE",
                    "COUNTRY": "SENDER_COUNTRY",
                    "ACCOUNT_TYPE": "SENDER_ACCOUNT_TYPE",
                    "TX_BEHAVIOR_ID": "SENDER_TX_BEHAVIOR_ID"
                }
            )

            # -------------------------------------------------
            # Merge sender
            # -------------------------------------------------

            merged = transactions.merge(
                sender_accounts,
                on="SENDER_ACCOUNT_ID",
                how="left",
                validate="many_to_one"
            )

            logging.info(
                f"After sender merge: {merged.shape}"
            )

            # -------------------------------------------------
            # Prepare RECEIVER account information
            # -------------------------------------------------

            receiver_accounts = accounts[
                [
                    "ACCOUNT_ID",
                    "CUSTOMER_ID",
                    "INIT_BALANCE",
                    "COUNTRY",
                    "ACCOUNT_TYPE",
                    "TX_BEHAVIOR_ID"
                ]
            ].copy()

            receiver_accounts = receiver_accounts.rename(
                columns={
                    "ACCOUNT_ID": "RECEIVER_ACCOUNT_ID",
                    "CUSTOMER_ID": "RECEIVER_CUSTOMER_ID",
                    "INIT_BALANCE": "RECEIVER_INIT_BALANCE",
                    "COUNTRY": "RECEIVER_COUNTRY",
                    "ACCOUNT_TYPE": "RECEIVER_ACCOUNT_TYPE",
                    "TX_BEHAVIOR_ID": "RECEIVER_TX_BEHAVIOR_ID"
                }
            )

            # -------------------------------------------------
            # Merge receiver
            # -------------------------------------------------

            merged = merged.merge(
                receiver_accounts,
                on="RECEIVER_ACCOUNT_ID",
                how="left",
                validate="many_to_one"
            )

            logging.info(
                f"After receiver merge: {merged.shape}"
            )

            return merged

        except Exception as e:
            logging.exception(
                "Error while merging account information."
            )
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # Validate merged data
    # ---------------------------------------------------------

    def validate_merged_data(self, merged):
        try:

            account_features = [
                "SENDER_CUSTOMER_ID",
                "SENDER_INIT_BALANCE",
                "SENDER_COUNTRY",
                "SENDER_ACCOUNT_TYPE",
                "SENDER_TX_BEHAVIOR_ID",
                "RECEIVER_CUSTOMER_ID",
                "RECEIVER_INIT_BALANCE",
                "RECEIVER_COUNTRY",
                "RECEIVER_ACCOUNT_TYPE",
                "RECEIVER_TX_BEHAVIOR_ID"
            ]

            missing_after_merge = (
                merged[account_features]
                .isnull()
                .sum()
            )

            if missing_after_merge.sum() > 0:

                logging.warning(
                    "Some account information could not be "
                    "matched during merge."
                )

                logging.warning(
                    missing_after_merge[
                        missing_after_merge > 0
                    ]
                )

            else:
                logging.info(
                    "All sender and receiver accounts "
                    "successfully matched."
                )

            # Check target
            if merged["IS_FRAUD"].isnull().any():
                raise ValueError(
                    "Transaction IS_FRAUD contains null values."
                )

            logging.info(
                "Merged dataset validation successful."
            )

        except Exception as e:
            logging.exception(
                "Merged dataset validation failed."
            )
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # Save intermediate dataset
    # ---------------------------------------------------------

    def save_merged_data(self, merged):
        try:

            output_path = (
                self.ingestion_config.merged_data_path
            )

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True
            )

            merged.to_csv(
                output_path,
                index=False
            )

            logging.info(
                f"Merged dataset saved at: {output_path}"
            )

            return output_path

        except Exception as e:
            logging.exception(
                "Failed to save merged dataset."
            )
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # Main ingestion method
    # ---------------------------------------------------------

    def initiate_data_ingestion(self):

        try:

            logging.info(
                "========== DATA INGESTION STARTED =========="
            )

            # 1. Load
            accounts, transactions, alerts = (
                self.load_data()
            )

            # 2. Validate columns
            self.validate_columns(
                accounts,
                transactions,
                alerts
            )

            # 3. Validate raw data
            self.validate_data(
                accounts,
                transactions,
                alerts
            )

            # 4. Merge account information
            merged_transactions = (
                self.merge_account_data(
                    accounts,
                    transactions
                )
            )

            # 5. Validate merged dataset
            self.validate_merged_data(
                merged_transactions
            )

            # 6. Save intermediate merged data
            merged_path = self.save_merged_data(
                merged_transactions
            )

            logging.info(
                "========== DATA INGESTION COMPLETED =========="
            )

            return (
                merged_transactions,
                alerts,
                merged_path
            )

        except Exception as e:

            logging.exception(
                "Data ingestion failed."
            )

            raise CustomException(e, sys)


# -------------------------------------------------------------
# Local testing
# -------------------------------------------------------------

if __name__ == "__main__":

    try:
        ingestion = DataIngestion()

        merged_data, alerts_data, path = (
            ingestion.initiate_data_ingestion()
        )

        print("\nMerged dataset shape:")
        print(merged_data.shape)

        print("\nMerged dataset columns:")
        print(merged_data.columns.tolist())

        print("\nFirst 5 rows:")
        print(merged_data.head())
    except Exception as e:
        logging.exception(
            "Data ingestion failed during local testing."
        )
        raise CustomException(e, sys)