import os
import sys

from dotenv import load_dotenv
from supabase import create_client, Client

from src.logger import logging
from src.exception import CustomException


load_dotenv()


class SupabaseClient:

    def __init__(self):
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")

            if not supabase_url or not supabase_key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_KEY "
                    "must be configured in the environment."
                )

            self.client: Client = create_client(
                supabase_url,
                supabase_key
            )

            logging.info(
                "Supabase client initialized successfully."
            )

        except Exception as e:
            raise CustomException(e, sys)

    def save_transaction(self, transaction, prediction):
        try:
            record = {
                "transaction_id": transaction["TX_ID"],
                "sender_account_id": transaction["SENDER_ACCOUNT_ID"],
                "receiver_account_id": transaction["RECEIVER_ACCOUNT_ID"],
                "transaction_type": transaction["TX_TYPE"],
                "transaction_amount": float(transaction["TX_AMOUNT"]),
                "transaction_timestamp": float(transaction["TIMESTAMP"]),
                "fraud_probability": float(
                    prediction["fraud_probability"]
                ),
                "fraud_prediction": int(
                    prediction["fraud_prediction"]
                ),
                "is_anomaly": bool(
                    prediction["is_anomaly"]
                ),
                "risk_score": float(
                    prediction["risk_score"]
                ),
                "decision": prediction["decision"]
            }

            response = (
                self.client
                .table("transactions")
                .insert(record)
                .execute()
            )

            logging.info(
                "Transaction prediction saved to Supabase: %s",
                transaction["TX_ID"]
            )

            return response.data

        except Exception as e:
            logging.exception(
                "Failed to save transaction to Supabase."
            )

            raise CustomException(e, sys)