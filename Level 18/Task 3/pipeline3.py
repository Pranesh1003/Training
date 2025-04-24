from pipelines.data_pipeline import simple_data_pipeline

if __name__ == "__main__":
    print("Starting ZenML pipeline execution...")
    run = simple_data_pipeline()  
    print("Pipeline execution completed!")
    print(f"Pipeline run ID: {run.id}")
