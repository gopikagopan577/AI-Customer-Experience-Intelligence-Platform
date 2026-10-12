
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from utils import load_numpy_data, print_data_shapes


# PART 1: LOAD CHURN DATA

X_train = load_numpy_data("data/X_train_churn.npy")
X_test = load_numpy_data("data/X_test_churn.npy")
y_train = load_numpy_data("data/y_train_churn.npy")
y_test = load_numpy_data("data/y_test_churn.npy")

print("Churn datasets loaded successfully!")

print_data_shapes(
    X_train,
    X_test,
    y_train,
    y_test
)

# Ensure the target is in the expected format
y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()


# PART 2: BUILD RANDOM FOREST MODEL

N_ESTIMATORS = 200
RANDOM_STATE = 42

churn_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# PART 3: TRAIN, EVALUATE AND TRACK WITH MLFLOW

mlflow.set_experiment("Customer_Churn_Prediction")

with mlflow.start_run(run_name="Churn_RandomForest_v1"):

    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": N_ESTIMATORS,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    })

# Train the model
    churn_model.fit(X_train, y_train)

# Predict churn on unseen test data
    y_pred = churn_model.predict(X_test)

# Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    print("\n" + "=" * 50)
    print("CUSTOMER CHURN MODEL RESULTS")
    print("=" * 50)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("Classification Report:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Churned", "Churned"],
        zero_division=0
    ))

# Log metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1_score": float(f1)
    })

# Display the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["Not Churned", "Churned"],
        cmap="Blues"
    )

    plt.title("Customer Churn Prediction - Confusion Matrix")
    plt.tight_layout()
    plt.show()
    plt.close()

# Log the trained model to MLflow
mlflow.sklearn.log_model(
    churn_model,
    name="churn_random_forest_model",
    serialization_format="pickle"
)

print("Churn metrics and model logged to MLflow.")


# PART 4: SAVE MODEL LOCALLY


os.makedirs("models", exist_ok=True)

joblib.dump(
    churn_model,
    "models/churn_model.pkl"
)

print("\nChurn model saved successfully!")
print("Location: models/churn_model.pkl")
print("Churn training and evaluation completed!")
