
import mlflow
import mlflow.sklearn
import joblib
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_name = "Customer_Segmentation_Model"
model_path = "models/segmentation_model.pkl"

# Load the existing trained model
model = joblib.load(model_path)

# Find the existing segmentation experiment
client = MlflowClient()
experiment = client.get_experiment_by_name(
    "Customer_Segmentation"
)

if experiment is None:
    experiment_id = client.create_experiment(
        "Customer_Segmentation"
    )
else:
    experiment_id = experiment.experiment_id

# Log the existing model and register it
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="Segmentation_Model_Registration"
):
    mlflow.sklearn.log_model(
        model,
        name="segmentation_kmeans_model",
        registered_model_name=model_name,
        serialization_format="pickle",
    )

print("Segmentation model registration completed.")
print("Model name:", model_name)
