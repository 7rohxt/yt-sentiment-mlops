import os
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    logger.info(f"Data loaded from {url}, shape = {df.shape}")
    return df

def save_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Raw data saved to {output_path}")

def main():
    try:
        params = load_params()
        cfg = params["data_ingestion"]

        url = cfg["source_url"]
        output_path = cfg["raw_data_path"]

        df = load_data(url)
        save_data(df, output_path)
        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.exception("Data ingestion failed.")
        raise e

if __name__ == "__main__":
    main()
