import os
import sys
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.logger import logging
from src.exception import CustomException


# =========================================================
# Save Object
# =========================================================

def save_object(file_path, obj):

    try:

        logging.info(
            f"Saving object to: {file_path}"
        )

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        joblib.dump(
            obj,
            file_path
        )

        logging.info(
            f"Object successfully saved: {file_path}"
        )

    except Exception as e:

        logging.exception(
            "Error while saving object."
        )

        raise CustomException(e, sys)


# =========================================================
# Load Object
# =========================================================

def load_object(file_path):

    try:

        logging.info(
            f"Loading object from: {file_path}"
        )

        obj = joblib.load(
            file_path
        )

        logging.info(
            f"Object successfully loaded: {file_path}"
        )

        return obj

    except Exception as e:

        logging.exception(
            "Error while loading object."
        )

        raise CustomException(e, sys)


# =========================================================
# Save NumPy Array
# =========================================================

def save_array(file_path, array):

    try:

        logging.info(
            f"Saving array to: {file_path}"
        )

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        np.save(
            file_path,
            array
        )

        logging.info(
            f"Array successfully saved."
        )

    except Exception as e:

        logging.exception(
            "Error while saving NumPy array."
        )

        raise CustomException(e, sys)


# =========================================================
# Load NumPy Array
# =========================================================

def load_array(file_path):

    try:

        logging.info(
            f"Loading array from: {file_path}"
        )

        array = np.load(
            file_path,
            allow_pickle=True
        )

        return array

    except Exception as e:

        logging.exception(
            "Error while loading NumPy array."
        )

        raise CustomException(e, sys)


# =========================================================
# Evaluate Classification Model
# =========================================================

def evaluate_classification_model(
    model,
    X_test,
    y_test
):

    try:

        logging.info(
            f"Evaluating model: "
            f"{model.__class__.__name__}"
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # -------------------------------------------------
        # Probability
        # -------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            y_probability = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

        elif hasattr(
            model,
            "decision_function"
        ):

            y_probability = (
                model.decision_function(
                    X_test
                )
            )

        else:

            y_probability = None

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # -------------------------------------------------
        # ROC-AUC
        # -------------------------------------------------

        if y_probability is not None:

            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )

            # -------------------------------------------------
            # PR-AUC
            # -------------------------------------------------

            pr_auc = average_precision_score(
                y_test,
                y_probability
            )

        else:

            roc_auc = None
            pr_auc = None

        metrics = {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": roc_auc,

            "pr_auc": pr_auc
        }

        logging.info(
            f"Model metrics: {metrics}"
        )

        return metrics

    except Exception as e:

        logging.exception(
            "Error while evaluating classification model."
        )

        raise CustomException(e, sys)