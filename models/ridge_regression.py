import os
import pandas as pd
import numpy as np
import wandb
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

TARGET_COLS = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
HORIZON_NAMES = ['24h', '48h', '72h']

def split_chronological(df: pd.DataFrame, train_ratio: float = 0.8):
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime').reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def train_ridge_model():
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        run = wandb.init(project="pearls-aqi-predictor", job_type="training-ridge")
    except Exception:
        run = wandb.init(project="pearls-aqi-predictor", job_type="training-ridge", mode="offline")
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_data = os.path.join(project_root, "data", "historical_aqi_features.csv")
    
    if os.path.exists(local_data):
        data_path = local_data
        print(f"Loading local multi-horizon features from: {data_path}")
    else:
        try:
            artifact = run.use_artifact('historical_aqi_features:latest')
            data_dir = artifact.download()
            data_path = os.path.join(data_dir, "historical_aqi_features.csv")
        except Exception as e:
            raise RuntimeError(f"Could not find historical data: {e}")
        
    df = pd.read_csv(data_path)
    
    # Chronological Split
    train_df, test_df = split_chronological(df, train_ratio=0.8)
    
    drop_cols = ['timestamp', 'datetime', 'city', 'country', 'aqi_target'] + TARGET_COLS
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df[TARGET_COLS]
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df[TARGET_COLS]
    
    feature_names = list(X_train.columns)
    
    print("Training Multi-Horizon Ridge Regression...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', MultiOutputRegressor(RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])))
    ])
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
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
        
        print(f"Ridge Horizon +{h_name} -> R2: {r2_h:.4f}, RMSE: {rmse_h:.4f}, MAE: {mae_h:.4f}")
        wandb.log({
            f"eval_ridge/{h_name}_r2": r2_h,
            f"eval_ridge/{h_name}_rmse": rmse_h,
            f"eval_ridge/{h_name}_mae": mae_h
        })
        
    avg_r2 = float(np.mean(r2_list))
    avg_rmse = float(np.mean(rmse_list))
    avg_mae = float(np.mean(mae_list))
    
    print(f"\nRidge Overall Multi-Horizon Summary - Mean R2: {avg_r2:.4f}, Mean RMSE: {avg_rmse:.4f}, Mean MAE: {avg_mae:.4f}")
    wandb.log({"model": "Ridge Regression", "mean_r2": avg_r2, "mean_rmse": avg_rmse, "mean_mae": avg_mae})
    
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_filepath = os.path.join(artifacts_dir, "ridge_regression_aqi.pkl")
    joblib.dump({
        'pipeline': pipeline,
        'feature_names': feature_names,
        'target_names': TARGET_COLS,
        'horizons': [24, 48, 72]
    }, model_filepath)
    
    model_artifact = wandb.Artifact(
        name="aqi_ridge_model",
        type="model",
        description=f"Multi-Horizon Ridge Regression (+24h R2={metrics['r2_24h']:.3f}, +48h R2={metrics['r2_48h']:.3f}, +72h R2={metrics['r2_72h']:.3f})"
    )
    model_artifact.add_file(model_filepath)
    run.log_artifact(model_artifact)
    run.finish()
    print("Multi-Horizon Ridge regression training complete!")
    return pipeline, {"r2": avg_r2, "rmse": avg_rmse, "mae": avg_mae, "horizon_metrics": metrics}

if __name__ == "__main__":
    train_ridge_model()
