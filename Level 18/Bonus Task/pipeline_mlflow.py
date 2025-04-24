import os
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.deploy_model import deploy_model

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"

@pipeline
def mlflow_deployment_pipeline():
    """
    A pipeline that ingests data, trains a model, and deploys it with MLflow.
    """
    data = ingest_data()
    
    model, metrics = train_model(data)
    
    model_uri = deploy_model(
        model=model, 
        metrics=metrics,
        model_name="zenml_linear_regression"
    )
    
    return model, metrics, model_uri

if __name__ == "__main__":

    mlflow_deployment_pipeline()
    
    print("\nPipeline complete! To view the MLflow UI, run:")
    print("mlflow ui")
    print("\nThen visit http://localhost:5000 in your browser")