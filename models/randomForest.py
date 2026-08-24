import os
import pandas as pd
import numpy as np
import wandb
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

TARGET_COLS = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
HORIZON_NAMES = ['24h', '48h', '72h']

def split_chronological(df: pd.DataFrame, train_ratio: float = 0.8):
    """Splits time series data chronologically across all cities to eliminate data leakage."""
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime').reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

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
        print(f"Loading 1-year multi-horizon features from: {data_path}")
    else:
        try:
            artifact = run.use_artifact('historical_aqi_features:latest')
            data_dir = artifact.download()
            data_path = os.path.join(data_dir, "historical_aqi_features.csv")
        except Exception as e:
            raise RuntimeError(f"Could not find historical data: {e}")
        
    df = pd.read_csv(data_path)
    
    # Verify multi-horizon targets exist
    if not all(col in df.columns for col in TARGET_COLS):
        from scripts.backfill_global_capitals import engineer_forecasting_features
        print("Transforming dataset into multi-horizon feature matrix...")
        df = engineer_forecasting_features(df)
        df.to_csv(data_path, index=False)
    
    # 2. Chronological Train-Test Split (Zero temporal leakage)
    print("Performing chronological train-test split (80% train, 20% test)...")
    train_df, test_df = split_chronological(df, train_ratio=0.8)
    
    drop_cols = ['timestamp', 'datetime', 'city', 'country', 'aqi_target'] + TARGET_COLS
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df[TARGET_COLS]
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df[TARGET_COLS]
    
    feature_names = list(X_train.columns)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}, Feature count: {len(feature_names)}")
    
    # 3. Train Multi-Horizon Random Forest Regressor
    print("Training Multi-Horizon Random Forest Regressor (+24h, +48h, +72h)...")
    model = RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate Horizon Performance Chronologically
    print("Evaluating multi-horizon out-of-time forecasting performance...")
    predictions = model.predict(X_test)
    
    metrics = {}
    r2_list, rmse_list, mae_list = [], [], []
    
    for idx, (target_col, h_name) in enumerate(zip(TARGET_COLS, HORIZON_NAMES)):
        y_true_h = y_test[target_col].values
        y_pred_h = predictions[:, idx]
        
        rmse_h = root_mean_squared_error(y_true_h, y_pred_h)
        mae_h = mean_absolute_error(y_true_h, y_pred_h)
        r2_h = r2_score(y_true_h, y_pred_h)
        
        r2_list.append(r2_h)
        rmse_list.append(rmse_h)
        mae_list.append(mae_h)
        
        metrics[f"r2_{h_name}"] = r2_h
        metrics[f"rmse_{h_name}"] = rmse_h
        metrics[f"mae_{h_name}"] = mae_h
        
        print(f"Horizon +{h_name} -> R2: {r2_h:.4f}, RMSE: {rmse_h:.4f}, MAE: {mae_h:.4f}")
        wandb.log({
            f"eval/{h_name}_r2": r2_h,
            f"eval/{h_name}_rmse": rmse_h,
            f"eval/{h_name}_mae": mae_h
        })
        
    avg_r2 = float(np.mean(r2_list))
    avg_rmse = float(np.mean(rmse_list))
    avg_mae = float(np.mean(mae_list))
    
    print(f"\nOverall Multi-Horizon Summary - Mean R2: {avg_r2:.4f}, Mean RMSE: {avg_rmse:.4f}, Mean MAE: {avg_mae:.4f}")
    wandb.log({"model": "Random Forest", "mean_r2": avg_r2, "mean_rmse": avg_rmse, "mean_mae": avg_mae})
    
    # Log Feature Importances
    feature_importances = dict(zip(feature_names, model.feature_importances_))
    for feat_name, importance in feature_importances.items():
        wandb.log({f"feature_importance/{feat_name}": importance})
        
    # Compute SHAP Values for Explainability
    print("Computing SHAP Explainability values...")
    try:
        sample_test = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_test)
        
        plt.figure(figsize=(10, 6))
        vals_24h = shap_values[:, :, 0] if len(shap_values.shape) == 3 else shap_values
        shap.summary_plot(vals_24h, sample_test, show=False)
        plt.title("SHAP Global Feature Importance Summary (+24h Horizon)")
        plt.tight_layout()
        
        shap_plot_path = "shap_summary_plot.png"
        plt.savefig(shap_plot_path, dpi=150)
        plt.close()
        
        wandb.log({"shap_summary": wandb.Image(shap_plot_path)})
        if os.path.exists(shap_plot_path):
            os.remove(shap_plot_path)
    except Exception as e:
        print(f"Warning: Could not generate SHAP plot: {e}")
        
    # 5. Save trained multi-horizon model (with compression)
    print("Saving multi-horizon model to artifacts folder & W&B Registry...")
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_filepath = os.path.join(artifacts_dir, "random_forest_aqi.pkl")
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'target_names': TARGET_COLS,
        'horizons': [24, 48, 72]
    }, model_filepath, compress=3)
    
    model_artifact = wandb.Artifact(
        name="aqi_random_forest_model",
        type="model",
        description=f"Multi-Horizon Random Forest Regressor (+24h R2={metrics['r2_24h']:.3f}, +48h R2={metrics['r2_48h']:.3f}, +72h R2={metrics['r2_72h']:.3f})"
    )
    model_artifact.add_file(model_filepath)
    run.log_artifact(model_artifact)
    
    run.finish()
    print("Multi-Horizon Random Forest training pipeline complete!")
    return model, {"r2": avg_r2, "rmse": avg_rmse, "mae": avg_mae, "horizon_metrics": metrics}

if __name__ == "__main__":
    train_model()