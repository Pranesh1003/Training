from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model

@pipeline
def model_training_pipeline():
    """
    A pipeline that ingests data and trains a model.
    """
    data = ingest_data()
    
    model = train_model(data)
    
    return model

if __name__ == "__main__":
    model_training_pipeline()