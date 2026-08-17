import os
import pandas as pd
import wandb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def train_ridge_model():
    run = wandb.init(project="pearls-aqi-predictor", job_type="training-ridge")
    print("Downloading historical data from W&B...")
    
    try:
        artifact = run.use_artifact('historical_aqi_features:latest')
        data_dir = artifact.download()
        data_path = os.path.join(data_dir, "historical_aqi_features.csv")
    except Exception as e:
        print(f"Warning: Could not fetch from W&B, checking local data: {e}")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(project_root, "data", "historical_aqi_features.csv")
        
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X = df.drop(columns=['timestamp', 'aqi_target'], errors='ignore')
    y = df['aqi_target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Ridge Regression with Cross-Validation...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))
    ])
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Ridge Model Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    wandb.log({"model": "Ridge Regression", "rmse": rmse, "mae": mae, "r2": r2})
    
    # Save model locally & to W&B
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_filepath = os.path.join(artifacts_dir, "ridge_regression_aqi.pkl")
    joblib.dump({
        'pipeline': pipeline,
        'feature_names': list(X.columns)
    }, model_filepath)
    
    model_artifact = wandb.Artifact(
        name="aqi_ridge_model",
        type="model",
        description="Ridge Regression model trained on Lahore AQI data"
    )
    model_artifact.add_file(model_filepath)
    run.log_artifact(model_artifact)
    run.finish()
    print("Ridge regression training complete!")
    return pipeline, {"rmse": rmse, "mae": mae, "r2": r2}

if __name__ == "__main__":
    train_ridge_model()
