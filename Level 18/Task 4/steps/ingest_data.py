import pandas as pd
from zenml import step

@step
def ingest_data() -> pd.DataFrame:
    """
    Ingests data for the pipeline.
    
    Returns:
        pandas DataFrame with the loaded data
    """
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': [2, 4, 6, 8, 10]
    }
    
    df = pd.DataFrame(data)
    print(f"Data ingested: {len(df)} rows, {len(df.columns)} columns")
    return df