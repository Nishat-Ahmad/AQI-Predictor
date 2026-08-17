import os
import pandas as pd
import wandb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def train_model():
    # 1. Initialize W&B and download the historical data
    # Notice we omitted the 'entity' parameter so it uses your default ghost-210 account
    run = wandb.init(project="pearls-aqi-predictor", job_type="training")
    print("Downloading historical data from W&B...")
    
    # Download the dataset we created in the last step
    artifact = run.use_artifact('historical_aqi_features:latest')
    data_dir = artifact.download()
    
    # Load the data into Pandas
    df = pd.read_csv(f"{data_dir}/historical_aqi_features.csv")
    
    # 2. Prepare the data for training
    print("Preparing data...")
    # Drop the timestamp since machine learning models need numbers, not strings
    X = df.drop(columns=['timestamp', 'aqi_target']) 
    y = df['aqi_target']
    
    # Split into 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train the Random Forest Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate the Model
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    
    # Calculate performance metrics
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Log metrics and feature importances to W&B dashboard
    wandb.log({"rmse": rmse, "mae": mae, "r2": r2})
    
    feature_importances = dict(zip(X.columns, model.feature_importances_))
    for feat_name, importance in feature_importances.items():
        wandb.log({f"feature_importance/{feat_name}": importance})
    print("Feature Importances:", feature_importances)
    
    # 5. Save the trained model to the local weights folder & Model Registry
    print("Saving model to local weights folder & W&B Model Registry...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(project_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    model_filepath = os.path.join(weights_dir, "random_forest_aqi.pkl")
    joblib.dump(model, model_filepath)
    
    # Create a Model Artifact
    model_artifact = wandb.Artifact(
        name="aqi_predictor_model",
        type="model",
        description="Random Forest Regressor trained on 2 years of Lahore AQI data"
    )
    model_artifact.add_file(model_filepath)
    run.log_artifact(model_artifact)
    
    run.finish()
    print("Training pipeline complete!")

if __name__ == "__main__":
    train_model()