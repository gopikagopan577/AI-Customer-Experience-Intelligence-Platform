
import mlflow
import mlflow.sklearn
import joblib
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_name = "Customer_Churn_Model"
model_path = "models/churn_model.pkl"

# Load the existing trained model
model = joblib.load(model_path)

# Use the existing churn experiment
client = MlflowClient()
experiment = client.get_experiment_by_name(
    "Customer_Churn_Prediction"
)

if experiment is None:
    experiment_id = client.create_experiment(
        "Customer_Churn_Prediction"
    )
else:
    experiment_id = experiment.experiment_id

# Log the existing model and register it
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="Churn_Model_Registration"
):
    mlflow.sklearn.log_model(
        model,
        name="churn_random_forest_model",
        registered_model_name=model_name,
        serialization_format="pickle",
    )

print("Churn model registration completed.")
print("Model name:", model_name)
