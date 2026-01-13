import mlflow

from week_5.load_index import index

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("LlamaIndex")

mlflow.models.set_model(index)