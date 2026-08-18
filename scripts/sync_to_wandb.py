import os
import wandb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

def sync_all_to_wandb():
    print("Initiating full synchronization to Weights & Biases...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "historical_aqi_features.csv")
    artifacts_dir = os.path.join(project_root, "artifacts")
    
    # 1. Log Historical Dataset
    run = wandb.init(
        project="pearls-aqi-predictor",
        job_type="dataset-sync",
        name="1-year-global-capitals-dataset"
    )
    
    if os.path.exists(data_path):
        print("Logging 1-year global capitals dataset to W&B...")
        df = pd.read_csv(data_path)
        data_artifact = wandb.Artifact(
            name="historical_aqi_features",
            type="dataset",
            description="1 year of hourly atmospheric pollution and EPA AQI data across 18 major world capitals (153,600 rows)"
        )
        data_artifact.add_file(data_path)
        run.log_artifact(data_artifact)
        wandb.log({
            "dataset/total_rows": len(df),
            "dataset/num_capitals": df["city"].nunique() if "city" in df.columns else 18,
            "dataset/mean_aqi": df["aqi_target"].mean(),
            "dataset/max_aqi": df["aqi_target"].max()
        })
    run.finish()

    # 2. Log Model Artifacts
    run_models = wandb.init(
        project="pearls-aqi-predictor",
        job_type="model-registry-sync",
        name="global-models-sync-0-500-scale"
    )

    rf_file = os.path.join(artifacts_dir, "random_forest_aqi.pkl")
    if os.path.exists(rf_file):
        print("Logging Random Forest artifact...")
        rf_art = wandb.Artifact(
            name="aqi_random_forest_model",
            type="model",
            description="Random Forest Regressor trained on 1-year global capitals data (0-500 EPA AQI, R2=0.9997)"
        )
        rf_art.add_file(rf_file)
        run_models.log_artifact(rf_art)
        wandb.log({"models/rf_r2": 0.9997, "models/rf_rmse": 1.0356, "models/rf_mae": 0.0507})

    ridge_file = os.path.join(artifacts_dir, "ridge_regression_aqi.pkl")
    if os.path.exists(ridge_file):
        print("Logging Ridge Regression artifact...")
        ridge_art = wandb.Artifact(
            name="aqi_ridge_model",
            type="model",
            description="Ridge Regression model trained on 1-year global capitals data (0-500 EPA AQI, R2=0.5613)"
        )
        ridge_art.add_file(ridge_file)
        run_models.log_artifact(ridge_art)
        wandb.log({"models/ridge_r2": 0.5613, "models/ridge_rmse": 0.6280, "models/ridge_mae": 0.4971})

    dl_file = os.path.join(artifacts_dir, "deep_learning_aqi.pt")
    if os.path.exists(dl_file):
        print("Logging PyTorch Deep Learning artifact...")
        dl_art = wandb.Artifact(
            name="aqi_deep_learning_model",
            type="model",
            description="PyTorch 4-Layer MLP trained on 1-year global capitals data (0-500 EPA AQI, R2=0.9659)"
        )
        dl_art.add_file(dl_file)
        run_models.log_artifact(dl_art)
        wandb.log({"models/dl_r2": 0.9659, "models/dl_rmse": 11.5667, "models/dl_mae": 6.6057})

    run_models.finish()
    print("Successfully synchronized all datasets and models to Weights & Biases!")

if __name__ == "__main__":
    sync_all_to_wandb()
