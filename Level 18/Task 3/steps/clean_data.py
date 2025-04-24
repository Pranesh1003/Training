from typing import Tuple
import pandas as pd
import numpy as np
from zenml import step

@step
def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]: 
    df = pd.DataFrame(X)
    df["dummy_column"] = np.nan
    df = df.drop(columns=["dummy_column"])
    df = df.fillna(df.median(numeric_only=True))
    return df, y
