import os
import sys
import joblib
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:

    model_dir: str = os.path.join(
        "artifacts",
        "models"
    )

    best_model_path: str = os.path.join(
        "artifacts",
        "models",
        "best_model.pkl"
    )

    isolation_forest_path: str = os.path.join(
        "artifacts",
        "models",
        "isolation_forest.pkl"
    )

    model_report_path: str = os.path.join(
        "artifacts",
        "models",
        "model_report.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

        os.makedirs(
            self.model_trainer_config.model_dir,
            exist_ok=True
        )

    # =====================================================
    # Load transformed data
    # =====================================================

    def load_transformed_data(self):

        try:

            logging.info(
                "Loading transformed train/test data."
            )

            train_data = joblib.load(
                os.path.join(
                    "artifacts",
                    "train_data.pkl"
                )
            )

            test_data = joblib.load(
                os.path.join(
                    "artifacts",
                    "test_data.pkl"
                )
            )

            X_train, y_train = train_data
            X_test, y_test = test_data

            logging.info(
                f"X_train shape: {X_train.shape}"
            )

            logging.info(
                f"X_test shape: {X_test.shape}"
            )

            logging.info(
                f"Training fraud count: {np.sum(y_train)}"
            )

            logging.info(
                f"Testing fraud count: {np.sum(y_test)}"
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:

            logging.exception(
                "Failed to load transformed data."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Create candidate models
    # =====================================================

    def get_models(self, y_train):

        try:

            negative_count = np.sum(y_train == 0)
            positive_count = np.sum(y_train == 1)

            if positive_count == 0:

                raise ValueError(
                    "Training data contains no fraud samples."
                )

            scale_pos_weight = (
                negative_count / positive_count
            )

            logging.info(
                f"scale_pos_weight: "
                f"{scale_pos_weight:.2f}"
            )

            models = {

                "LogisticRegression":
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear"
                    ),

                "RandomForest":
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42
                    ),

                "ExtraTrees":
                    ExtraTreesClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42
                    ),

                "XGBoost":
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=-1,
                        random_state=42
                    ),

                "HistGradientBoosting":
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        random_state=42
                    )
            }

            return models

        except Exception as e:

            logging.exception(
                "Failed to create models."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Evaluate model
    # =====================================================

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        try:

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):

                y_probability = (
                    model.predict_proba(X_test)[:, 1]
                )

            elif hasattr(model, "decision_function"):

                y_probability = (
                    model.decision_function(X_test)
                )

            else:

                y_probability = y_pred

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

            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )

            pr_auc = average_precision_score(
                y_test,
                y_probability
            )

            return {

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1": f1,

                "roc_auc": roc_auc,

                "pr_auc": pr_auc
            }

        except Exception as e:

            logging.exception(
                "Model evaluation failed."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Train supervised models
    # =====================================================

    def train_supervised_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        try:

            models = self.get_models(
                y_train
            )

            results = {}

            best_model = None
            best_model_name = None
            best_pr_auc = -np.inf

            logging.info(
                "========== SUPERVISED MODEL TRAINING STARTED =========="
            )

            for model_name, model in models.items():

                logging.info(
                    f"Training {model_name}..."
                )

                model.fit(
                    X_train,
                    y_train
                )

                metrics = self.evaluate_model(
                    model,
                    X_test,
                    y_test
                )

                results[model_name] = metrics

                logging.info(
                    f"{model_name} | "
                    f"Accuracy={metrics['accuracy']:.4f} | "
                    f"Precision={metrics['precision']:.4f} | "
                    f"Recall={metrics['recall']:.4f} | "
                    f"F1={metrics['f1']:.4f} | "
                    f"ROC-AUC={metrics['roc_auc']:.4f} | "
                    f"PR-AUC={metrics['pr_auc']:.4f}"
                )

                # Best model = highest PR-AUC
                if metrics["pr_auc"] > best_pr_auc:

                    best_pr_auc = metrics["pr_auc"]

                    best_model = model

                    best_model_name = model_name

            logging.info(
                f"Best model: {best_model_name} "
                f"with PR-AUC={best_pr_auc:.4f}"
            )

            return (
                best_model,
                best_model_name,
                results
            )

        except Exception as e:

            logging.exception(
                "Supervised model training failed."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Isolation Forest
    # =====================================================

    def train_isolation_forest(
        self,
        X_train
    ):

        try:

            logging.info(
                "Training Isolation Forest."
            )

            isolation_model = IsolationForest(
                n_estimators=300,
                contamination="auto",
                random_state=42,
                n_jobs=-1
            )

            isolation_model.fit(
                X_train
            )

            joblib.dump(
                isolation_model,
                self.model_trainer_config.isolation_forest_path
            )

            logging.info(
                "Isolation Forest saved."
            )

            return isolation_model

        except Exception as e:

            logging.exception(
                "Isolation Forest training failed."
            )

            raise CustomException(e, sys)

    # =====================================================
    # Main method
    # =====================================================

    def initiate_model_trainer(self):

        try:

            logging.info(
                "========== MODEL TRAINER STARTED =========="
            )

            # 1. Load data

            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = self.load_transformed_data()

            # 2. Train supervised models

            (
                best_model,
                best_model_name,
                results
            ) = self.train_supervised_models(
                X_train,
                X_test,
                y_train,
                y_test
            )

            # 3. Save best model

            joblib.dump(
                best_model,
                self.model_trainer_config.best_model_path
            )

            # 4. Train anomaly model

            isolation_model = (
                self.train_isolation_forest(
                    X_train
                )
            )

            # 5. Save report

            model_report = {

                "best_model": best_model_name,

                "selection_metric": "pr_auc",

                "supervised_results": results
            }

            joblib.dump(
                model_report,
                self.model_trainer_config.model_report_path
            )

            logging.info(
                "========== MODEL TRAINER COMPLETED =========="
            )

            return (
                best_model,
                best_model_name,
                results,
                isolation_model
            )

        except Exception as e:

            logging.exception(
                "Model training failed."
            )

            raise CustomException(e, sys)


# =========================================================
# Local testing
# =========================================================

if __name__ == "__main__":

    try:

        trainer = ModelTrainer()

        (
            best_model,
            best_model_name,
            results,
            isolation_model
        ) = trainer.initiate_model_trainer()

        print("\n========================================")
        print("MODEL TRAINING COMPLETED")
        print("========================================")

        print(
            f"\nBest model: {best_model_name}"
        )

        print("\nModel comparison:")

        for model_name, metrics in results.items():

            print(f"\n{model_name}")

            print(
                f"  Accuracy : "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"  Precision: "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"  Recall   : "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"  F1       : "
                f"{metrics['f1']:.4f}"
            )

            print(
                f"  ROC-AUC  : "
                f"{metrics['roc_auc']:.4f}"
            )

            print(
                f"  PR-AUC   : "
                f"{metrics['pr_auc']:.4f}"
            )

        print(
            "\nBest model saved at:"
        )

        print(
            trainer.model_trainer_config.best_model_path
        )

        print(
            "\nIsolation Forest saved at:"
        )

        print(
            trainer.model_trainer_config.isolation_forest_path
        )

        print(
            "\nModel report saved at:"
        )

        print(
            trainer.model_trainer_config.model_report_path
        )

    except Exception as e:

        logging.exception(
            "Model training failed during local testing."
        )

        raise CustomException(e, sys)