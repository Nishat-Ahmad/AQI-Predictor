import os
import torch
import joblib
import numpy as np

def export_pytorch_to_numpy():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pt_path = os.path.join(project_root, "artifacts", "deep_learning_aqi.pt")
    out_path = os.path.join(project_root, "artifacts", "deep_learning_numpy.pkl")
    
    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} does not exist.")
        return
    
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    scaler = checkpoint.get("scaler")
    feature_names = checkpoint.get("feature_names")
    input_dim = checkpoint.get("input_dim", 11)
    
    numpy_weights = {}
    for k, v in state_dict.items():
        numpy_weights[k] = v.cpu().numpy()
        print(f"Extracted {k}: shape {v.shape}")
        
    payload = {
        "weights": numpy_weights,
        "scaler": scaler,
        "feature_names": feature_names,
        "input_dim": input_dim
    }
    
    joblib.dump(payload, out_path)
    print(f"Successfully exported pure NumPy Deep Learning weights to: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    export_pytorch_to_numpy()
