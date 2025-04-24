import zenml
from zenml import pipeline, step
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple
from model_dev import LinearRegressionModel, RandomForestModel, ModelEvaluator

@step
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Generate sample regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@step
def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    model.train(X_train, y_train)
    return model

@step
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    return metrics

@pipeline
def model_comparison_pipeline():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train and evaluate Linear Regression
    lr_model = LinearRegressionModel()
    trained_lr = train_model(lr_model, X_train, y_train)
    lr_metrics = evaluate_model(trained_lr, X_test, y_test)
    
    # Train and evaluate Random Forest
    rf_model = RandomForestModel()
    trained_rf = train_model(rf_model, X_train, y_train)
    rf_metrics = evaluate_model(trained_rf, X_test, y_test)
    
    # Combine metrics for comparison
    comparison = {
        'Linear Regression': lr_metrics,
        'Random Forest': rf_metrics
    }
    
    return comparison

if __name__ == "__main__":
    # Run the pipeline
    pipeline = model_comparison_pipeline()
    pipeline.run()