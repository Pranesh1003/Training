import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from zenml import step
from typing import Dict

@step
def evaluate_model(
    model: LinearRegression,
    data: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained LinearRegression model
        data: DataFrame containing test data
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model performance...")

    X = data.iloc[:, :-1]  
    y_true = data.iloc[:, -1] 
    
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "r2_score": r2
    }
    print(f"Model Evaluation Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return metrics