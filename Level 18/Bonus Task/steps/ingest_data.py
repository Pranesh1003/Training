import pandas as pd
from zenml import step

@step
def ingest_data() -> pd.DataFrame:
    """
    Ingests data for the pipeline.
    
    Returns:
        pandas DataFrame with the loaded data
    """
    print("Ingesting data...")
    
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'target': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    }
    
    df = pd.DataFrame(data)
    print(f"Data ingested: {len(df)} rows, {len(df.columns)} columns")
    return df