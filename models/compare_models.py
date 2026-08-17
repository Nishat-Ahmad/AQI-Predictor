import os
import pandas as pd
import numpy as np
import wandb
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt

# PyTorch Deep Learning Model Architecture
class AQINeuralNet(nn.Module):
    def __init__(self, input_dim):
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
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def load_data(run):
    try:
        artifact = run.use_artifact('historical_aqi_features:latest')
        data_dir = artifact.download()
        data_path = os.path.join(data_dir, "historical_aqi_features.csv")
    except Exception as e:
        print(f"Warning: Could not fetch artifact from W&B, falling back to local: {e}")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(project_root, "data", "historical_aqi_features.csv")
        
    df = pd.read_csv(data_path)
    return df

def run_model_comparison():
    print("=" * 60)
    print("AQI PREDICTOR: UNIFIED MODEL BENCHMARKING & TRAINING")
    print("=" * 60)
    
    run = wandb.init(project="pearls-aqi-predictor", job_type="model-comparison-benchmark")
    df = load_data(run)
    
    X = df.drop(columns=['timestamp', 'aqi_target'], errors='ignore')
    y = df['aqi_target']
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    leaderboard = []
    
    # ----------------------------------------------------
    # 1. Ridge Regression
    # ----------------------------------------------------
    print("\n[1/3] Training Ridge Regression...")
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))
    ])
    ridge_pipeline.fit(X_train, y_train)
    ridge_preds = ridge_pipeline.predict(X_test)
    
    ridge_metrics = {
        "Model": "Ridge Regression",
        "RMSE": root_mean_squared_error(y_test, ridge_preds),
        "MAE": mean_absolute_error(y_test, ridge_preds),
        "R2": r2_score(y_test, ridge_preds)
    }
    leaderboard.append(ridge_metrics)
    
    ridge_file = os.path.join(artifacts_dir, "ridge_regression_aqi.pkl")
    joblib.dump({'pipeline': ridge_pipeline, 'feature_names': feature_names}, ridge_file)
    
    art_ridge = wandb.Artifact("aqi_ridge_model", type="model")
    art_ridge.add_file(ridge_file)
    run.log_artifact(art_ridge)
    
    # ----------------------------------------------------
    # 2. Random Forest Regressor
    # ----------------------------------------------------
    print("\n[2/3] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_metrics = {
        "Model": "Random Forest",
        "RMSE": root_mean_squared_error(y_test, rf_preds),
        "MAE": mean_absolute_error(y_test, rf_preds),
        "R2": r2_score(y_test, rf_preds)
    }
    leaderboard.append(rf_metrics)
    
    rf_file = os.path.join(artifacts_dir, "random_forest_aqi.pkl")
    joblib.dump({'model': rf_model, 'feature_names': feature_names}, rf_file)
    
    art_rf = wandb.Artifact("aqi_predictor_model", type="model")
    art_rf.add_file(rf_file)
    run.log_artifact(art_rf)
    
    # ----------------------------------------------------
    # 3. Deep Learning PyTorch MLP
    # ----------------------------------------------------
    print("\n[3/3] Training Deep Learning (PyTorch MLP)...")
    dl_scaler = StandardScaler()
    X_train_scaled = dl_scaler.fit_transform(X_train)
    X_test_scaled = dl_scaler.transform(X_test)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_model = AQINeuralNet(input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.003, weight_decay=1e-4)
    
    dl_model.train()
    for epoch in range(1, 36):
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
        dl_preds = dl_model(test_x_t).cpu().numpy().flatten()
        
    dl_metrics = {
        "Model": "Deep Learning (PyTorch MLP)",
        "RMSE": root_mean_squared_error(y_test, dl_preds),
        "MAE": mean_absolute_error(y_test, dl_preds),
        "R2": r2_score(y_test, dl_preds)
    }
    leaderboard.append(dl_metrics)
    
    dl_file = os.path.join(artifacts_dir, "deep_learning_aqi.pt")
    torch.save({
        'state_dict': dl_model.state_dict(),
        'input_dim': X.shape[1],
        'feature_names': feature_names,
        'scaler': dl_scaler
    }, dl_file)
    
    art_dl = wandb.Artifact("aqi_deep_learning_model", type="model")
    art_dl.add_file(dl_file)
    run.log_artifact(art_dl)
    
    # ----------------------------------------------------
    # Leaderboard & W&B Summary Table
    # ----------------------------------------------------
    summary_df = pd.DataFrame(leaderboard).sort_values("RMSE").reset_index(drop=True)
    print("\n" + "=" * 60)
    print("BENCHMARK LEADERBOARD (Sorted by lowest RMSE):")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)
    
    # Log Table to W&B
    wb_table = wandb.Table(dataframe=summary_df)
    run.log({"Model Benchmark Leaderboard": wb_table})
    
    # Generate SHAP Global Summary Plot
    try:
        sample_test = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer(sample_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, sample_test, show=False)
        plt.title("SHAP Feature Importance (Random Forest)")
        plt.tight_layout()
        shap_img = "shap_comparison_plot.png"
        plt.savefig(shap_img, dpi=150)
        plt.close()
        run.log({"SHAP Global Summary": wandb.Image(shap_img)})
        if os.path.exists(shap_img):
            os.remove(shap_img)
    except Exception as e:
        print(f"SHAP logging note: {e}")
        
    run.finish()
    print("\nModel comparison complete! All 3 models saved and logged to W&B.")

if __name__ == "__main__":
    run_model_comparison()
