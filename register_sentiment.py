
import mlflow
from mlflow import MlflowClient

# Connect to your existing local MLflow tracking store
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "Customer_Sentiment_Analysis"
)

if experiment is None:
    raise ValueError("Sentiment experiment not found.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="",
    order_by=["attribute.start_time DESC"],
)

# Find the selected run by its exact historical name
selected_run = next(
    (
        run for run in runs
        if run.info.run_name == "LSTM_Sentiment_ClassWeighted"
    ),
    None,
)

if selected_run is None:
    raise ValueError(
        "Selected run not found. Check the run name in MLflow."
    )

run_id = selected_run.info.run_id

# Register the logged model artifact
model_uri = f"runs:/{run_id}/sentiment_lstm_model"

model_name = "Customer_Sentiment_Model"

registered = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

print("Registered model:", model_name)
print("Model version:", registered.version)
print("Source run ID:", run_id)
