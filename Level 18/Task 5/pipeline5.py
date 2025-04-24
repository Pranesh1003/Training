from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def model_evaluation_pipeline():
    """
    A pipeline that ingests data, trains a model, and evaluates its performance.
    """
    data = ingest_data()

    model = train_model(data)

    metrics = evaluate_model(model=model, data=data)
    
    return model, metrics

if __name__ == "__main__":

    model_evaluation_pipeline()