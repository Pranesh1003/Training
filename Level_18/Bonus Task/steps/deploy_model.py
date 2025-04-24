import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from zenml import step
from typing import Dict, Any, Optional

@step
def deploy_model(
    model: LinearRegression,
    metrics: Dict[str, Any],
    model_name: str = "zenml_linear_regression"
) -> str:
    """
    Deploy the trained model to MLflow model registry.
    
    Args:
        model: Trained LinearRegression model
        metrics: Dictionary of model metrics
        model_name: Name to register the model with in MLflow
        
    Returns:
        Model URI for the deployed model
    """
    print(f"Deploying model to MLflow model registry as '{model_name}'...")
    
    run_id = metrics.get("mlflow_run_id")
    
    if not run_id:
       
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_metrics(metrics)
            model_uri = mlflow.sklearn.log_model(
                model,
                "deployed_model",
                registered_model_name=model_name
            ).model_uri
    else:

        model_uri = f"runs:/{run_id}/linear_regression_model"
        
        client = MlflowClient()
        
    
        try:
            model_version = client.get_latest_versions(model_name, stages=["None"])[0].version
            print(f"Model '{model_name}' already exists with version {model_version}")
        except:
            
            client.create_registered_model(model_name)
            result = mlflow.register_model(model_uri, model_name)
            print(f"Registered model '{model_name}' as version {result.version}")
            
    print(f"Model deployed successfully. Model URI: {model_uri}")
    return model_uri