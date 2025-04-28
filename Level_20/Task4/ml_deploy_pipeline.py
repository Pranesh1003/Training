import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from zenml import pipeline, step
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
import uuid
from typing import Tuple, Optional
import mlflow

@step
def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

@step
def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

@step
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

@step
def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("r2_score", r2) 
    return r2

@step
def deployment_trigger(r2_score: float) -> bool:
    return r2_score >= 0.5

@step
def log_model(model: RandomForestRegressor, X_train: pd.DataFrame) -> None:
    input_example = X_train[:1] 
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

@pipeline
def deployment_pipeline():
    mlflow.set_experiment("california_housing_deployment")  

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    r2 = evaluate_model(model, X_test, y_test)
    log_model(model, X_train)
    should_deploy = deployment_trigger(r2)
    if should_deploy:
        mlflow_model_deployer_step(
            model_name=f"california_housing_model_{uuid.uuid4().hex}",
            model=model,
            deploy_decision=should_deploy
        )

if __name__ == "__main__":
    deployment_pipeline()