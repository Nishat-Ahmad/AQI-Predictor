import os
import pandas as pd
import numpy as np
import wandb
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt

TARGET_COLS = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
HORIZON_NAMES = ['24h', '48h', '72h']

class AQINeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(AQINeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def load_data(run):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_data = os.path.join(project_root, "data", "historical_aqi_features.csv")
    
    if os.path.exists(local_data):
        df = pd.read_csv(local_data)
    else:
        try:
            artifact = run.use_artifact('historical_aqi_features:latest')
            data_dir = artifact.download()
            df = pd.read_csv(os.path.join(data_dir, "historical_aqi_features.csv"))
        except Exception as e:
            raise RuntimeError(f"Could not load data: {e}")
            
    return df

def split_chronological(df: pd.DataFrame, train_ratio: float = 0.8):
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime').reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def run_model_comparison():
    print("=" * 70)
    print("AQI PREDICTOR: MULTI-HORIZON (24h/48h/72h) MODEL BENCHMARKING")
    print("=" * 70)
    
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        run = wandb.init(project="pearls-aqi-predictor", job_type="multi-horizon-benchmark")
    except Exception:
        run = wandb.init(project="pearls-aqi-predictor", job_type="multi-horizon-benchmark", mode="offline")
        
    df = load_data(run)
    train_df, test_df = split_chronological(df, train_ratio=0.8)
    
    drop_cols = ['timestamp', 'datetime', 'city', 'country', 'aqi_target'] + TARGET_COLS
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df[TARGET_COLS]
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df[TARGET_COLS]
    
    feature_names = list(X_train.columns)
    print(f"Data Split -> Train: {len(X_train)} samples, Test (Out-of-time): {len(X_test)} samples, Features: {len(feature_names)}")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    leaderboard = []
    
    # ----------------------------------------------------
    # 1. Multi-Horizon Ridge Regression
    # ----------------------------------------------------
    print("\n[1/3] Training Multi-Horizon Ridge Regression...")
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', MultiOutputRegressor(RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])))
    ])
    ridge_pipeline.fit(X_train, y_train)
    ridge_preds = ridge_pipeline.predict(X_test)
    
    ridge_r2_24 = r2_score(y_test['target_aqi_24h'], ridge_preds[:, 0])
    ridge_r2_48 = r2_score(y_test['target_aqi_48h'], ridge_preds[:, 1])
    ridge_r2_72 = r2_score(y_test['target_aqi_72h'], ridge_preds[:, 2])
    
    ridge_rmse_24 = root_mean_squared_error(y_test['target_aqi_24h'], ridge_preds[:, 0])
    ridge_mae_24 = mean_absolute_error(y_test['target_aqi_24h'], ridge_preds[:, 0])
    
    ridge_metrics = {
        "Model": "Ridge Regression",
        "R2 (24h)": round(ridge_r2_24, 4),
        "R2 (48h)": round(ridge_r2_48, 4),
        "R2 (72h)": round(ridge_r2_72, 4),
        "Mean R2": round(float(np.mean([ridge_r2_24, ridge_r2_48, ridge_r2_72])), 4),
        "RMSE (24h)": round(ridge_rmse_24, 4),
        "MAE (24h)": round(ridge_mae_24, 4)
    }
    leaderboard.append(ridge_metrics)
    
    ridge_file = os.path.join(artifacts_dir, "ridge_regression_aqi.pkl")
    joblib.dump({'pipeline': ridge_pipeline, 'feature_names': feature_names, 'target_names': TARGET_COLS, 'horizons': [24, 48, 72]}, ridge_file)
    
    art_ridge = wandb.Artifact("aqi_ridge_model", type="model")
    art_ridge.add_file(ridge_file)
    run.log_artifact(art_ridge)
    
    # ----------------------------------------------------
    # 2. Multi-Horizon Random Forest Regressor
    # ----------------------------------------------------
    print("\n[2/3] Training Multi-Horizon Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_r2_24 = r2_score(y_test['target_aqi_24h'], rf_preds[:, 0])
    rf_r2_48 = r2_score(y_test['target_aqi_48h'], rf_preds[:, 1])
    rf_r2_72 = r2_score(y_test['target_aqi_72h'], rf_preds[:, 2])
    
    rf_rmse_24 = root_mean_squared_error(y_test['target_aqi_24h'], rf_preds[:, 0])
    rf_mae_24 = mean_absolute_error(y_test['target_aqi_24h'], rf_preds[:, 0])
    
    rf_metrics = {
        "Model": "Random Forest",
        "R2 (24h)": round(rf_r2_24, 4),
        "R2 (48h)": round(rf_r2_48, 4),
        "R2 (72h)": round(rf_r2_72, 4),
        "Mean R2": round(float(np.mean([rf_r2_24, rf_r2_48, rf_r2_72])), 4),
        "RMSE (24h)": round(rf_rmse_24, 4),
        "MAE (24h)": round(rf_mae_24, 4)
    }
    leaderboard.append(rf_metrics)
    
    rf_file = os.path.join(artifacts_dir, "random_forest_aqi.pkl")
    joblib.dump({'model': rf_model, 'feature_names': feature_names, 'target_names': TARGET_COLS, 'horizons': [24, 48, 72]}, rf_file, compress=3)
    
    art_rf = wandb.Artifact("aqi_random_forest_model", type="model")
    art_rf.add_file(rf_file)
    run.log_artifact(art_rf)
    
    # ----------------------------------------------------
    # 3. Deep Learning PyTorch MLP
    # ----------------------------------------------------
    print("\n[3/3] Training Multi-Horizon Deep Learning (PyTorch MLP)...")
    dl_scaler = StandardScaler()
    X_train_scaled = dl_scaler.fit_transform(X_train)
    X_test_scaled = dl_scaler.transform(X_test)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_model = AQINeuralNet(input_dim=X_train.shape[1], output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    dl_model.train()
    for epoch in range(1, 31):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = dl_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    dl_model.eval()
    with torch.no_grad():
        test_x_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        dl_preds = dl_model(test_x_t).cpu().numpy()
        
    dl_r2_24 = r2_score(y_test['target_aqi_24h'], dl_preds[:, 0])
    dl_r2_48 = r2_score(y_test['target_aqi_48h'], dl_preds[:, 1])
    dl_r2_72 = r2_score(y_test['target_aqi_72h'], dl_preds[:, 2])
    
    dl_rmse_24 = root_mean_squared_error(y_test['target_aqi_24h'], dl_preds[:, 0])
    dl_mae_24 = mean_absolute_error(y_test['target_aqi_24h'], dl_preds[:, 0])
    
    dl_metrics = {
        "Model": "Deep Learning (PyTorch MLP)",
        "R2 (24h)": round(dl_r2_24, 4),
        "R2 (48h)": round(dl_r2_48, 4),
        "R2 (72h)": round(dl_r2_72, 4),
        "Mean R2": round(float(np.mean([dl_r2_24, dl_r2_48, dl_r2_72])), 4),
        "RMSE (24h)": round(dl_rmse_24, 4),
        "MAE (24h)": round(dl_mae_24, 4)
    }
    leaderboard.append(dl_metrics)
    
    # Save PyTorch .pt
    dl_file = os.path.join(artifacts_dir, "deep_learning_aqi.pt")
    torch.save({
        'state_dict': dl_model.state_dict(),
        'input_dim': X_train.shape[1],
        'output_dim': 3,
        'feature_names': feature_names,
        'target_names': TARGET_COLS,
        'horizons': [24, 48, 72],
        'scaler': dl_scaler
    }, dl_file)
    
    # Save pure NumPy format
    numpy_filepath = os.path.join(artifacts_dir, "deep_learning_numpy.pkl")
    numpy_payload = {
        "weights": {k: v.cpu().numpy() for k, v in dl_model.state_dict().items()},
        "scaler": dl_scaler,
        "feature_names": feature_names,
        "target_names": TARGET_COLS,
        "horizons": [24, 48, 72],
        "input_dim": X_train.shape[1],
        "output_dim": 3
    }
    joblib.dump(numpy_payload, numpy_filepath)
    
    art_dl = wandb.Artifact("aqi_deep_learning_model", type="model")
    art_dl.add_file(dl_file)
    run.log_artifact(art_dl)
    
    # ----------------------------------------------------
    # Leaderboard Summary
    # ----------------------------------------------------
    summary_df = pd.DataFrame(leaderboard).sort_values("Mean R2", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 80)
    print("TRUE MULTI-HORIZON FORECASTING LEADERBOARD (Out-of-Time Test Set):")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    wb_table = wandb.Table(dataframe=summary_df)
    run.log({"Multi-Horizon Leaderboard": wb_table})
    
    # SHAP Global Summary Plot (+24h horizon)
    try:
        sample_test = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer(sample_test)
        
        plt.figure(figsize=(10, 6))
        vals_24h = shap_vals[:, :, 0] if len(shap_vals.shape) == 3 else shap_vals
        shap.summary_plot(vals_24h, sample_test, show=False)
        plt.title("SHAP Feature Importance (Random Forest +24h Forecast)")
        plt.tight_layout()
        shap_img = "shap_comparison_plot.png"
        plt.savefig(shap_img, dpi=150)
        plt.close()
        run.log({"SHAP Global Summary": wandb.Image(shap_img)})
        if os.path.exists(shap_img):
            os.remove(shap_img)
    except Exception as e:
        print(f"SHAP note: {e}")
        
    run.finish()
    print("\nMulti-Horizon model benchmarking complete! All models saved and logged.")
    return summary_df

if __name__ == "__main__":
    run_model_comparison()
