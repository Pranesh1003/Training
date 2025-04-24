import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"

def test_deployed_model():
    """Test the deployed model with sample data."""
    print("Testing deployed model from MLflow...")

    test_data = {
        'feature1': [1.5, 2.5, 3.5],
        'feature2': [15, 25, 35],
        'feature3': [0.15, 0.25, 0.35]
    }
    test_df = pd.DataFrame(test_data)

    client = MlflowClient()
    model_name = "zenml_linear_regression"
    
    try:
        latest_model = client.get_latest_versions(model_name, stages=["None"])[0]
        model_uri = f"models:/{model_name}/{latest_model.version}"
        print(f"Using model: {model_name}, version: {latest_model.version}")
        
     
        model = mlflow.sklearn.load_model(model_uri)
        
        predictions = model.predict(test_df)
        
        result_df = test_df.copy()
        result_df['predicted_target'] = predictions
        
        print("\nTest results:")
        print(result_df)
        
        return result_df
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've run the pipeline first and the model is registered in MLflow.")
        return None

if __name__ == "__main__":
    test_deployed_model()