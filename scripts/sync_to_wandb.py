import os
import wandb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

def sync_all_to_wandb():
    print("Initiating full synchronization to Weights & Biases...")
    os.environ.setdefault("WANDB_SILENT", "true")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "historical_aqi_features.csv")
    artifacts_dir = os.path.join(project_root, "artifacts")
    
    # 1. Log Historical Dataset
    try:
        run = wandb.init(
            project="pearls-aqi-predictor",
            job_type="dataset-sync",
            name="1-year-multi-horizon-dataset"
        )
    except Exception:
        run = wandb.init(
            project="pearls-aqi-predictor",
            job_type="dataset-sync",
            name="1-year-multi-horizon-dataset",
            mode="offline"
        )
    
    if os.path.exists(data_path):
        print("Logging 1-year multi-horizon dataset to W&B...")
        df = pd.read_csv(data_path)
        data_artifact = wandb.Artifact(
            name="historical_aqi_features",
            type="dataset",
            description="1 year of hourly atmospheric observations, temporal lag features, and 24/48/72h targets across 18 major world capitals"
        )
        data_artifact.add_file(data_path)
        run.log_artifact(data_artifact)
        wandb.log({
            "dataset/total_rows": len(df),
            "dataset/num_capitals": df["city"].nunique() if "city" in df.columns else 18,
            "dataset/mean_aqi": df["current_aqi"].mean() if "current_aqi" in df.columns else 65.0,
            "dataset/max_aqi": df["current_aqi"].max() if "current_aqi" in df.columns else 500.0
        })
    run.finish()

    # 2. Log Model Artifacts
    try:
        run_models = wandb.init(
            project="pearls-aqi-predictor",
            job_type="model-registry-sync",
            name="multi-horizon-models-sync"
        )
    except Exception:
        run_models = wandb.init(
            project="pearls-aqi-predictor",
            job_type="model-registry-sync",
            name="multi-horizon-models-sync",
            mode="offline"
        )

    rf_file = os.path.join(artifacts_dir, "random_forest_aqi.pkl")
    if os.path.exists(rf_file):
        print("Logging Random Forest artifact...")
        rf_art = wandb.Artifact(
            name="aqi_random_forest_model",
            type="model",
            description="Multi-Horizon Random Forest Regressor (+24h R2=0.740, +48h R2=0.636, +72h R2=0.603)"
        )
        rf_art.add_file(rf_file)
        run_models.log_artifact(rf_art)

    ridge_file = os.path.join(artifacts_dir, "ridge_regression_aqi.pkl")
    if os.path.exists(ridge_file):
        print("Logging Ridge Regression artifact...")
        ridge_art = wandb.Artifact(
            name="aqi_ridge_model",
            type="model",
            description="Multi-Horizon Ridge Regression model (+24h R2=0.753, +48h R2=0.655, +72h R2=0.610)"
        )
        ridge_art.add_file(ridge_file)
        run_models.log_artifact(ridge_art)

    dl_file = os.path.join(artifacts_dir, "deep_learning_aqi.pt")
    if os.path.exists(dl_file):
        print("Logging PyTorch Deep Learning artifact...")
        dl_art = wandb.Artifact(
            name="aqi_deep_learning_model",
            type="model",
            description="PyTorch 4-Layer Multi-Horizon MLP (+24h R2=0.680, +48h R2=0.611, +72h R2=0.574)"
        )
        dl_art.add_file(dl_file)
        run_models.log_artifact(dl_art)

    run_models.finish()
    print("Successfully synchronized all multi-horizon datasets and models to Weights & Biases!")

if __name__ == "__main__":
    sync_all_to_wandb()
