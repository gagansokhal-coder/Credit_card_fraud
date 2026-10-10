import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.logger import logging
from src.database.supabase_client import SupabaseClient
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Detect potentially fraudulent financial transactions.",
    version="1.0.0"
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


class TransactionRequest(BaseModel):
    TX_ID: str
    SENDER_ACCOUNT_ID: str
    RECEIVER_ACCOUNT_ID: str

    TX_TYPE: str = "TRANSFER"

    TX_AMOUNT: float = Field(gt=0)
    TIMESTAMP: float

    SENDER_CUSTOMER_ID: str
    RECEIVER_CUSTOMER_ID: str

    SENDER_COUNTRY: str
    RECEIVER_COUNTRY: str

    SENDER_ACCOUNT_TYPE: str
    RECEIVER_ACCOUNT_TYPE: str

    SENDER_INIT_BALANCE: float = Field(ge=0)
    RECEIVER_INIT_BALANCE: float = Field(ge=0)

    SENDER_TX_BEHAVIOR_ID: int
    RECEIVER_TX_BEHAVIOR_ID: int


predictor = None
database = None

@app.on_event("startup")
def load_predictor():
    global predictor, database

    try:
        predictor = PredictPipeline()
        database = SupabaseClient()

        logging.info("Fraud prediction API initialized successfully.")
        logging.info("Supabase database client initialized successfully.")

    except Exception:
        logging.exception("Failed to initialize the application.")
        raise


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={}
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
        "predictor_loaded": predictor is not None
    }

@app.get("/transactions")
def get_transactions():
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not initialized."
        )

    try:
        response = (
            database.client
            .table("transactions")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )

        return response.data

    except Exception:
        logging.exception("Failed to retrieve transactions.")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve transactions."
        )
@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not initialized."
        )

    try:
        response = (
            database.client
            .table("transactions")
            .select("*")
            .eq("transaction_id", transaction_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Transaction not found."
            )

        return response.data[0]

    except HTTPException:
        raise

    except Exception:
        logging.exception("Failed to retrieve transaction details.")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve transaction details."
        )
@app.get("/stats")
def get_stats():
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not initialized."
        )

    try:
        client = database.client

        total = (
            client.table("transactions")
            .select("id", count="exact", head=True)
            .execute()
        )

        fraud = (
            client.table("transactions")
            .select("id", count="exact", head=True)
            .eq("fraud_prediction", 1)
            .execute()
        )

        approved = (
            client.table("transactions")
            .select("id", count="exact", head=True)
            .eq("decision", "APPROVE")
            .execute()
        )

        reviewed = (
            client.table("transactions")
            .select("id", count="exact", head=True)
            .eq("decision", "REVIEW")
            .execute()
        )

        blocked = (
            client.table("transactions")
            .select("id", count="exact", head=True)
            .eq("decision", "BLOCK")
            .execute()
        )

        return {
            "total_transactions": total.count or 0,
            "fraud_predictions": fraud.count or 0,
            "approve_decisions": approved.count or 0,
            "review_decisions": reviewed.count or 0,
            "block_decisions": blocked.count or 0
        }

    except Exception:
        logging.exception("Failed to calculate dashboard statistics.")

        raise HTTPException(
            status_code=500,
            detail="Failed to calculate dashboard statistics."
        )
@app.post("/predict")
def predict_transaction(transaction: TransactionRequest):
    if predictor is None or database is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction pipeline or database is not initialized."
        )

    try:
        transaction_data = transaction.model_dump()

        # Generate fraud prediction
        result = predictor.predict(transaction_data)

        # Save transaction and prediction to Supabase
        saved_record = database.save_transaction(
            transaction_data,
            result
        )

        logging.info(
            "Prediction generated and transaction saved successfully."
        )

        return {
            "transaction_id": transaction.TX_ID,
            "prediction": result,
            "database_saved": bool(saved_record)
        }

    except Exception:
        logging.exception("Transaction prediction or database save failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction or database saving failed. Check application logs."
        )