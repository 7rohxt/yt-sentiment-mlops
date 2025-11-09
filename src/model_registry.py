import mlflow
import logging
import yaml
from dotenv import load_dotenv
import os

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_registry")

# --- Load environment variables ---
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    try:
        params = load_params()
        model_name = "FinalModel"

        # --- Get latest run from Model_Building experiment ---
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("Final_Model")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        logger.info(f"Registering model from run_id={run_id}")

        # --- Register the model ---
        model_uri = f"runs:/{run_id}/LinearSVC_model"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.info(f"Model registered as '{model_name}' (version {result.version}) in MLflow registry.")
    except Exception as e:
        logger.exception("Model registration failed.")
        raise e

if __name__ == "__main__":
    main()
