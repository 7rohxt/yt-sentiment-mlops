import os
import yaml
import pandas as pd
import joblib
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

# --- Load env and configure MLflow ---
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Final_Model_Evaluation")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_evaluation")

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    try:
        params = load_params()
        data_path = params["data_preprocessing"]["processed_data_path"]

        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded processed data from {data_path}")

        # Load artifacts
        model = joblib.load("models/linearsvc_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_comment"], df["category"],
            test_size=params["data_preprocessing"]["test_size"],
            random_state=params["model_building"]["random_state"]
        )

        # Transform test set
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        logger.info(f"✅ Evaluation complete | Acc={acc:.4f}, F1={f1:.4f}")

        # Plot and save confusion matrix
        os.makedirs("reports", exist_ok=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("LinearSVC Confusion Matrix")
        plt.savefig("reports/conf_matrix.png")

        # Save numeric report
        report_path = "reports/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })
            mlflow.log_artifact("reports/conf_matrix.png")
            mlflow.log_artifact(report_path)
            mlflow.log_param("model_type", "LinearSVC")
            mlflow.log_param("vectorizer", "TF-IDF")

    except Exception as e:
        logger.exception("Evaluation failed.")
        raise e

if __name__ == "__main__":
    main()
