import mlflow
import logging
from dotenv import load_dotenv
import os

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_registry")

# --- Load env vars ---
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


def main():
    try:
        model_name = "FinalModel"
        client = mlflow.tracking.MlflowClient()

        exp = client.get_experiment_by_name("Final_Model")
        if exp is None:
            raise ValueError("Experiment 'Final_Model' not found in MLflow tracking server.")

        # --- Get latest run ---
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        logger.info(f"Registering model from run_id={run_id}")

        # --- Register the unified pipeline ---
        model_uri = f"runs:/{run_id}/LinearSVC_pipeline"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.info(f"✅ Registered '{model_name}' (version {result.version}) in MLflow registry.")
    except Exception as e:
        logger.exception("Model registration failed.")
        raise e


if __name__ == "__main__":
    main()
