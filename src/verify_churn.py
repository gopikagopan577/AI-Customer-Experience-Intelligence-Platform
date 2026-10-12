
import joblib
import numpy as np

# Load the saved churn model
model = joblib.load("models/churn_model.pkl")

# Load test data
X_test = np.load("data/X_test_churn.npy")
y_test = np.load("data/y_test_churn.npy")

# Predict a few customers
predictions = model.predict(X_test[:5])

print("Churn model loaded successfully!")
print("Model type:", type(model).__name__)
print("Test data shape:", X_test.shape)
print("Actual churn labels:", y_test[:5])
print("Predicted churn labels:", predictions)
