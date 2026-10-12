
import os
import numpy as np
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Embedding,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from utils import load_numpy_data, print_data_shapes


# PART 1: LOAD SENTIMENT DATA

X_train = load_numpy_data("data/X_train_sentiment.npy")
X_test = load_numpy_data("data/X_test_sentiment.npy")
y_train = load_numpy_data("data/y_train_sentiment.npy")
y_test = load_numpy_data("data/y_test_sentiment.npy")

print("Sentiment datasets loaded successfully!")
print_data_shapes(X_train, X_test, y_train, y_test)


# PART 2: BUILD LSTM MODEL

VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
LSTM_UNITS = 32
DROPOUT_RATE = 0.3
EPOCHS = 5
BATCH_SIZE = 64

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),
    LSTM(LSTM_UNITS),
    Dropout(DROPOUT_RATE),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# PART 3: CALCULATE CLASS WEIGHTS

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("Class weights:", class_weights)


# PART 4: TRAIN, EVALUATE AND TRACK WITH MLFLOW

mlflow.set_experiment("Customer_Sentiment_Analysis")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

with mlflow.start_run(run_name="Sentiment_LSTM_ClassWeighted_v1"):

    mlflow.log_params({
        "model_type": "LSTM",
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "class_weighting": "balanced",
        "max_length": 100
    })

# Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )

# Evaluate on the held-out test dataset
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    y_probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("Sentiment Model Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")

    print("Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    ))

# Log evaluation metrics in this same MLflow run
    mlflow.log_metrics({
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1)
    })

# Record per-class precision, recall and F1
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Negative", "Neutral", "Positive"],
        output_dict=True,
        zero_division=0
    )

    for class_name in ["Negative", "Neutral", "Positive"]:
        mlflow.log_metrics({
            f"{class_name.lower()}_precision":
                float(report[class_name]["precision"]),
            f"{class_name.lower()}_recall":
                float(report[class_name]["recall"]),
            f"{class_name.lower()}_f1_score":
                float(report[class_name]["f1-score"])
        })

    
# PART 5: VISUAL MODEL EVALUATION

# Graph 1: Training and validation curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )
    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )
    axes[0].set_title("LSTM Accuracy Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history.history["loss"],
        label="Training Loss"
    )
    axes[1].plot(
        history.history["val_loss"],
        label="Validation Loss"
    )
    axes[1].set_title("LSTM Loss Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()
    plt.close(fig)

# Graph 2: Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )

    class_names = ["Negative", "Neutral", "Positive"]

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Sentiment",
        ylabel="Actual Sentiment",
        title="Sentiment Analysis Confusion Matrix"
    )

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print("Confusion Matrix:")
    print(cm)

# Log the trained model in this same MLflow run
    mlflow.tensorflow.log_model(
        model,
        name="sentiment_lstm_model"
    )

    print("Metrics and model logged to MLflow.")


# PART 6: SAVE MODEL LOCALLY

os.makedirs("models", exist_ok=True)

model.save("models/sentiment_model.keras")

print("Sentiment model saved successfully!")
print("Location: models/sentiment_model.keras")
print("Sentiment model training and evaluation completed!")
