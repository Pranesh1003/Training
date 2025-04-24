from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.datasets import load_digits


@step
def load_digits_data() -> Tuple[np.ndarray, np.ndarray]:
    digits = load_digits()
    return digits.data, digits.target


@step
def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(X)
    df["dummy_column"] = np.nan
    df = df.drop(columns=["dummy_column"])
    df = df.fillna(df.median(numeric_only=True))
    return df, y
