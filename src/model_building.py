import os
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import logging
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- MLflow setup ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Final_Model")

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_building")


def load_params(path="params.yaml"):
    """Load parameters from YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    try:
        params = load_params()
        cfg = params["model_building"]
        data_cfg = params["data_preprocessing"]

        # --- Load preprocessed data ---
        df = pd.read_csv(data_cfg["processed_data_path"])
        logger.info(f"Loaded preprocessed data from {data_cfg['processed_data_path']} | Shape: {df.shape}")

        # --- Train-test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_comment"],
            df["category"],
            test_size=data_cfg["test_size"],
            random_state=cfg["random_state"]
        )

        # --- Define and train pipeline ---
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=cfg["max_features"])),
            ("model", LinearSVC(**cfg["model_params"], random_state=cfg["random_state"]))
        ])

        logger.info("Training TF-IDF + LinearSVC pipeline...")
        pipeline.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        logger.info(f"✅ Model trained | Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        # --- MLflow Logging ---
        with mlflow.start_run(run_name="LinearSVC_Pipeline"):
            mlflow.log_params(cfg["model_params"])
            mlflow.log_param("max_features", cfg["max_features"])
            mlflow.log_param("random_state", cfg["random_state"])
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

            # Log entire pipeline (no test input example)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="LinearSVC_pipeline"#,
                # registered_model_name="FinalModel"
            )

        # --- Save locally for DVC tracking ---
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/linearsvc_pipeline.pkl")
        logger.info("Saved unified pipeline under 'models/linearsvc_pipeline.pkl'")

    except Exception as e:
        logger.exception("Model building failed.")
        raise e


if __name__ == "__main__":
    main()
