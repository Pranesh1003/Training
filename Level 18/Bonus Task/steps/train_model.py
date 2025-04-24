import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from zenml import step
from typing import Dict, Any, Tuple

@step
def train_model(df: pd.DataFrame) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train a Linear Regression model and log it with MLflow.
    
    Args:
        df: Input pandas DataFrame with features and target
        
    Returns:
        Tuple containing:
        - Trained LinearRegression model
        - Dictionary of model metrics
    """
    print("Starting model training with MLflow tracking...")
    
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1]   
    
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
 
    with mlflow.start_run() as run:
 
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("features", feature_names)
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
       
        mlflow.log_metric("train_r2", train_score)
        mlflow.log_metric("test_r2", test_score)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        
        mlflow.sklearn.log_model(
            model, 
            "linear_regression_model",
            registered_model_name="zenml_linear_regression"
        )
        
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")
        
    print(f"Model training complete and logged to MLflow")
    print(f"R² on training data: {train_score:.4f}")
    print(f"R² on test data: {test_score:.4f}")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    
    metrics = {
        "train_r2": train_score,
        "test_r2": test_score,
        "mse": mse,
        "rmse": rmse,
        "mlflow_run_id": run_id
    }
    
    return model, metrics