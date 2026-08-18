import os
import pandas as pd
import wandb
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def train_model():
    # 1. Initialize W&B
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        run = wandb.init(project="pearls-aqi-predictor", job_type="training-random-forest")
    except Exception:
        run = wandb.init(project="pearls-aqi-predictor", job_type="training-random-forest", mode="offline")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_data = os.path.join(project_root, "data", "historical_aqi_features.csv")
    if os.path.exists(local_data):
        data_path = local_data
        print(f"Loading local 1-year global features from: {data_path}")
    else:
        try:
            artifact = run.use_artifact('historical_aqi_features:latest')
            data_dir = artifact.download()
            data_path = os.path.join(data_dir, "historical_aqi_features.csv")
        except Exception as e:
            raise RuntimeError(f"Could not find historical data: {e}")
        
    df = pd.read_csv(data_path)
    
    # 2. Prepare data for training
    print("Preparing data...")
    X = df.drop(columns=['timestamp', 'aqi_target', 'city', 'country'], errors='ignore')
    y = df['aqi_target']
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Random Forest Regressor
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate Model Performance
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Random Forest Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    wandb.log({"model": "Random Forest", "rmse": rmse, "mae": mae, "r2": r2})
    
    feature_importances = dict(zip(X.columns, model.feature_importances_))
    for feat_name, importance in feature_importances.items():
        wandb.log({f"feature_importance/{feat_name}": importance})
        
    # Compute SHAP Values for Explainability
    print("Computing SHAP Explainability values...")
    try:
        sample_test = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_test, show=False)
        plt.title("SHAP Global Feature Importance Summary")
        plt.tight_layout()
        
        shap_plot_path = "shap_summary_plot.png"
        plt.savefig(shap_plot_path, dpi=150)
        plt.close()
        
        wandb.log({"shap_summary": wandb.Image(shap_plot_path)})
        if os.path.exists(shap_plot_path):
            os.remove(shap_plot_path)
    except Exception as e:
        print(f"Warning: Could not generate SHAP plot: {e}")
        
    # 5. Save trained model
    print("Saving model to local artifacts folder & W&B Model Registry...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_filepath = os.path.join(artifacts_dir, "random_forest_aqi.pkl")
    joblib.dump({
        'model': model,
        'feature_names': feature_names
    }, model_filepath)
    
    model_artifact = wandb.Artifact(
        name="aqi_predictor_model",
        type="model",
        description="Random Forest Regressor trained on 2 years of Lahore AQI data"
    )
    model_artifact.add_file(model_filepath)
    run.log_artifact(model_artifact)
    
    run.finish()
    print("Random Forest training pipeline complete!")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}

if __name__ == "__main__":
    train_model()