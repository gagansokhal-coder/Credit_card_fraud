import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):

        try:

            logging.info(
                "================================================"
            )

            logging.info(
                "========== TRAINING PIPELINE STARTED =========="
            )

            logging.info(
                "================================================"
            )

            # =================================================
            # STEP 1: DATA INGESTION
            # =================================================

            logging.info(
                "STEP 1: Starting data ingestion."
            )

            data_ingestion = DataIngestion()

            (
                merged_data,
                alerts_data,
                merged_data_path
            ) = data_ingestion.initiate_data_ingestion()

            logging.info(
                "STEP 1 COMPLETED: Data ingestion."
            )

            logging.info(
                f"Merged data path: {merged_data_path}"
            )

            # =================================================
            # STEP 2: DATA TRANSFORMATION
            # =================================================

            logging.info(
                "STEP 2: Starting data transformation."
            )

            data_transformation = DataTransformation()

            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = (
                data_transformation
                .initiate_data_transformation(
                    merged_data
                )
            )

            logging.info(
                "STEP 2 COMPLETED: Data transformation."
            )

            logging.info(
                f"X_train shape: {X_train.shape}"
            )

            logging.info(
                f"X_test shape: {X_test.shape}"
            )

            # =================================================
            # STEP 3: MODEL TRAINING
            # =================================================

            logging.info(
                "STEP 3: Starting model training."
            )

            model_trainer = ModelTrainer()

            (
                best_model,
                best_model_name,
                model_results,
                isolation_model
            ) = (
                model_trainer
                .initiate_model_trainer()
            )

            logging.info(
                "STEP 3 COMPLETED: Model training."
            )

            logging.info(
                f"Best model: {best_model_name}"
            )

            # =================================================
            # PIPELINE COMPLETED
            # =================================================

            logging.info(
                "================================================"
            )

            logging.info(
                "========== TRAINING PIPELINE COMPLETED ========"
            )

            logging.info(
                "================================================"
            )

            return {
                "best_model": best_model_name,
                "model_results": model_results
            }

        except Exception as e:

            logging.exception(
                "Training pipeline failed."
            )

            raise CustomException(e, sys)


# =========================================================
# Local Testing
# =========================================================

if __name__ == "__main__":

    try:

        pipeline = TrainPipeline()

        results = pipeline.run_pipeline()

        print("\n========================================")
        print("TRAINING PIPELINE COMPLETED")
        print("========================================")

        print(
            f"\nBest Model: "
            f"{results['best_model']}"
        )

        print("\nModel Results:")

        for model_name, metrics in (
            results["model_results"].items()
        ):

            print(
                f"\n{model_name}"
            )

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

    except Exception as e:

        logging.exception(
            "Training pipeline failed during execution."
        )

        raise CustomException(e, sys)