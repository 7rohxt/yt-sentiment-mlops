import os
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import logging
from dotenv import load_dotenv

load_dotenv() 

# --- MLflow setup ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Final_Model")

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_building")

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    try:
        params = load_params()
        cfg = params["model_building"]

        # --- Load preprocessed data ---
        data_path = params["data_preprocessing"]["processed_data_path"]
        df = pd.read_csv(data_path)
        logger.info(f"Loaded preprocessed data from {data_path} | Shape: {df.shape}")

        # --- Train-test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_comment"],
            df["category"],
            test_size=params["data_preprocessing"]["test_size"],
            random_state=cfg["random_state"]
        )

        # --- TF-IDF Vectorization ---
        logger.info("Applying TF-IDF vectorization...")
        vectorizer = TfidfVectorizer(max_features=cfg["max_features"])
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # --- LinearSVC Training ---
        model = LinearSVC(**cfg["model_params"], random_state=cfg["random_state"])
        logger.info("Training LinearSVC model...")
        model.fit(X_train_vec, y_train)

        # --- Evaluation ---
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        logger.info(f"✅ Training complete | Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        # --- MLflow Tracking ---
        with mlflow.start_run():
            mlflow.log_params(cfg["model_params"])
            mlflow.log_param("max_features", cfg["max_features"])
            mlflow.log_param("random_state", cfg["random_state"])
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
            mlflow.sklearn.log_model(model, artifact_path="LinearSVC_model")

        # --- Save artifacts locally ---
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/linearsvc_model.pkl")
        joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
        logger.info("Model and vectorizer saved under 'models/'")

    except Exception as e:
        logger.exception("Model building failed.")
        raise e

if __name__ == "__main__":
    main()
