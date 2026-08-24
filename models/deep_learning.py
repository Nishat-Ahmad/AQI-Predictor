import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import wandb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

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

def split_chronological(df: pd.DataFrame, train_ratio: float = 0.8):
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime').reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def train_deep_learning_model():
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        run = wandb.init(project="pearls-aqi-predictor", job_type="training-deep-learning")
    except Exception:
        run = wandb.init(project="pearls-aqi-predictor", job_type="training-deep-learning", mode="offline")
        
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
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    print("Initializing Multi-Horizon PyTorch AQI Neural Network (3 outputs: +24h, +48h, +72h)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AQINeuralNet(input_dim=X_train.shape[1], output_dim=3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 30
    print(f"Training on {device} for {epochs} epochs...")
    
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        scheduler.step(epoch_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Loss (MSE): {epoch_loss:.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        predictions = model(test_x_tensor).cpu().numpy()
        
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
        
        print(f"Deep Learning Horizon +{h_name} -> R2: {r2_h:.4f}, RMSE: {rmse_h:.4f}, MAE: {mae_h:.4f}")
        wandb.log({
            f"eval_dl/{h_name}_r2": r2_h,
            f"eval_dl/{h_name}_rmse": rmse_h,
            f"eval_dl/{h_name}_mae": mae_h
        })
        
    avg_r2 = float(np.mean(r2_list))
    avg_rmse = float(np.mean(rmse_list))
    avg_mae = float(np.mean(mae_list))
    
    print(f"\nDeep Learning Overall Multi-Horizon Summary - Mean R2: {avg_r2:.4f}, Mean RMSE: {avg_rmse:.4f}, Mean MAE: {avg_mae:.4f}")
    wandb.log({"model": "Deep Learning (PyTorch MLP)", "mean_r2": avg_r2, "mean_rmse": avg_rmse, "mean_mae": avg_mae})
    
    # Save model artifacts
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_filepath = os.path.join(artifacts_dir, "deep_learning_aqi.pt")
    payload = {
        'state_dict': model.state_dict(),
        'input_dim': X_train.shape[1],
        'output_dim': 3,
        'feature_names': feature_names,
        'target_names': TARGET_COLS,
        'horizons': [24, 48, 72],
        'scaler': scaler
    }
    torch.save(payload, model_filepath)
    
    # Save pure NumPy format for serverless production deployment (zero torch dependency)
    numpy_filepath = os.path.join(artifacts_dir, "deep_learning_numpy.pkl")
    numpy_payload = {
        "weights": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "feature_names": feature_names,
        "target_names": TARGET_COLS,
        "horizons": [24, 48, 72],
        "input_dim": X_train.shape[1],
        "output_dim": 3
    }
    joblib.dump(numpy_payload, numpy_filepath)
    print(f"Saved pure NumPy weights to: {numpy_filepath}")
    
    model_artifact = wandb.Artifact(
        name="aqi_deep_learning_model",
        type="model",
        description=f"PyTorch Multi-Horizon MLP (+24h R2={metrics['r2_24h']:.3f}, +48h R2={metrics['r2_48h']:.3f}, +72h R2={metrics['r2_72h']:.3f})"
    )
    model_artifact.add_file(model_filepath)
    run.log_artifact(model_artifact)
    run.finish()
    print("Multi-Horizon Deep Learning training complete!")
    return model, scaler, {"r2": avg_r2, "rmse": avg_rmse, "mae": avg_mae, "horizon_metrics": metrics}

if __name__ == "__main__":
    train_deep_learning_model()
