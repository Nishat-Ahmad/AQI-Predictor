import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import wandb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Define Deep Learning PyTorch Architecture
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
        print(f"Loading local 1-year global features from: {data_path}")
    else:
        try:
            artifact = run.use_artifact('historical_aqi_features:latest')
            data_dir = artifact.download()
            data_path = os.path.join(data_dir, "historical_aqi_features.csv")
        except Exception as e:
            raise RuntimeError(f"Could not find historical data: {e}")
        
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['timestamp', 'aqi_target', 'city', 'country'], errors='ignore')
    y = df['aqi_target']
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print("Initializing PyTorch AQI Neural Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AQINeuralNet(input_dim=X.shape[1]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 40
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
        predictions = model(test_x_tensor).cpu().numpy().flatten()
        
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Deep Learning Model Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    wandb.log({"model": "Deep Learning (PyTorch MLP)", "rmse": rmse, "mae": mae, "r2": r2})
    
    # Save model weights, architecture info, and scaler
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_filepath = os.path.join(artifacts_dir, "deep_learning_aqi.pt")
    payload = {
        'state_dict': model.state_dict(),
        'input_dim': X.shape[1],
        'feature_names': feature_names,
        'scaler': scaler
    }
    torch.save(payload, model_filepath)
    
    # Save pure NumPy format for serverless production deployment (zero torch dependency)
    numpy_filepath = os.path.join(artifacts_dir, "deep_learning_numpy.pkl")
    numpy_payload = {
        "weights": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "feature_names": feature_names,
        "input_dim": X.shape[1]
    }
    joblib.dump(numpy_payload, numpy_filepath)
    print(f"Saved pure NumPy weights to: {numpy_filepath}")
    
    model_artifact = wandb.Artifact(
        name="aqi_deep_learning_model",
        type="model",
        description="PyTorch Deep Learning MLP trained on Lahore AQI data"
    )
    model_artifact.add_file(model_filepath)
    run.log_artifact(model_artifact)
    run.finish()
    print("Deep Learning training complete!")
    return model, scaler, {"rmse": rmse, "mae": mae, "r2": r2}

if __name__ == "__main__":
    train_deep_learning_model()
