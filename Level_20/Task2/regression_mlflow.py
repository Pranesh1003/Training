import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from zenml import step, pipeline
from zenml.client import Client
import logging
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def ingest_data() -> pd.DataFrame:
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    logger.info("Data ingested successfully.")
    return df

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by removing null values."""
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Removed {initial_rows - len(df)} rows with null values.")
    return df

@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    mlflow.sklearn.autolog()
    
    X = df.drop('target', axis=1)
    y = df['target']
    model = LinearRegression()
    model.fit(X, y)
    logger.info("Model training completed with coefficients: %s", model.coef_)
    return model

@step
def evaluate_model(model: LinearRegression, df: pd.DataFrame) -> dict:
    X = df.drop('target', axis=1)
    y = df['target']
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    metrics = {"MSE": mse, "R2": r2}
    
    # Log metrics to MLflow
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)
    
    logger.info(f"Evaluation metrics: MSE={mse:.2f}, R2={r2:.2f}")
    return metrics

@pipeline
def regression_pipeline():
    data = ingest_data()
    cleaned_data = clean_data(data)
    model = train_model(cleaned_data)
    metrics = evaluate_model(model, cleaned_data)

if __name__ == "__main__":
    Client()
    
    regression_pipeline()