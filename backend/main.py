
import re
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore

from backend.schemas import (
    SentimentRequest,
    CustomerRequest,
    SegmentationRequest,
)


# 1. CREATE FASTAPI APPLICATION


app = FastAPI(
    title="AI-Powered Customer Experience Intelligence Platform",
    description=(
        "An API for customer sentiment analysis, "
        "churn prediction, and customer segmentation."
    ),
    version="1.0.0",
)



# 2. LOAD SAVED MODELS AND PREPROCESSING OBJECTS


sentiment_model = load_model("models/sentiment_model.keras")
sentiment_tokenizer = joblib.load("models/sentiment_tokenizer.pkl")
sentiment_label_encoder = joblib.load(
    "models/sentiment_label_encoder.pkl"
)

churn_model = joblib.load("models/churn_model.pkl")
churn_location_encoder = joblib.load(
    "models/churn_location_encoder.pkl"
)
churn_scaler = joblib.load("models/churn_scaler.pkl")
churn_feature_names = joblib.load(
    "models/churn_feature_names.pkl"
)

segmentation_model = joblib.load("models/segmentation_model.pkl")
segmentation_scaler = joblib.load(
    "models/segmentation_scaler.pkl"
)
segmentation_feature_names = joblib.load(
    "models/segmentation_feature_names.pkl"
)


# 3. FEATURE DEFINITIONS
# Must match src/preprocess.py

CHURN_NUMERICAL_FEATURES = [
    "age",
    "total_orders",
    "avg_order_value",
    "days_since_last_order",
    "support_tickets",
    "customer_satisfaction",
]

SEGMENTATION_FEATURES = [
    "total_orders",
    "avg_order_value",
    "days_since_last_order",
    "support_tickets",
    "customer_satisfaction",
]



# 4. TEXT CLEANING
# Must match sentiment preprocessing

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 5. HOME ENDPOINT

@app.get("/")
def home():
    return {
        "message": "Customer Experience Intelligence Platform API is running",
        "endpoints": [
            "/predict/sentiment",
            "/predict/churn",
            "/predict/segment",
            "/health",
        ],
    }


# 6. HEALTH CHECK

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": [
            "sentiment",
            "churn",
            "segmentation",
        ],
    }


# 7. SENTIMENT ANALYSIS

@app.post("/predict/sentiment")
def predict_sentiment(request: SentimentRequest):

    cleaned_review = clean_text(request.review_text)

    if not cleaned_review:
        raise HTTPException(
            status_code=422,
            detail="Please enter a review containing English letters.",
        )

    sequence = sentiment_tokenizer.texts_to_sequences(
        [cleaned_review]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=100,
        padding="post",
        truncating="post",
    )

    probabilities = sentiment_model.predict(
        padded_sequence,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_label = sentiment_label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return {
        "review_text": request.review_text,
        "predicted_sentiment": str(predicted_label),
        "confidence": round(
            float(probabilities[predicted_index]), 4
        ),
        "class_probabilities": {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                sentiment_label_encoder.classes_,
                probabilities,
            )
        },
    }



# 8. CUSTOMER CHURN PREDICTION

@app.post("/predict/churn")
def predict_churn(request: CustomerRequest):

    customer = request.model_dump()

    # Prepare numerical features in training order
    numerical_df = pd.DataFrame(
        [[customer[name] for name in CHURN_NUMERICAL_FEATURES]],
        columns=CHURN_NUMERICAL_FEATURES,
    )

    scaled_numerical = churn_scaler.transform(numerical_df)

    # Encode location using the saved encoder
    location_df = pd.DataFrame({
        "location": [customer["location"]]
    })

    encoded_location = churn_location_encoder.transform(
        location_df
    )

    location_columns = list(
        churn_location_encoder.get_feature_names_out(
            ["location"]
        )
    )

    # Combine numerical and location features
    processed_customer = pd.DataFrame(
        np.hstack([scaled_numerical, encoded_location]),
        columns=CHURN_NUMERICAL_FEATURES + location_columns,
    )

    # Restore exact feature order used during training
    processed_customer = processed_customer.reindex(
        columns=churn_feature_names,
        fill_value=0,
    )

    prediction = int(
        churn_model.predict(processed_customer)[0]
    )

    result = {
        "predicted_churn": prediction,
        "churn_status": (
            "Churn risk"
            if prediction == 1
            else "Not predicted to churn"
        ),
    }

    if hasattr(churn_model, "predict_proba"):
        probabilities = churn_model.predict_proba(
            processed_customer
        )[0]

        model_classes = list(churn_model.classes_)

        if 1 in model_classes:
            churn_probability = float(
                probabilities[model_classes.index(1)]
            )
            result["churn_probability"] = round(
                churn_probability, 4
            )

    return result


# 9. CUSTOMER SEGMENTATION

@app.post("/predict/segment")
def predict_segment(request: SegmentationRequest):

    customer = request.model_dump()

    customer_df = pd.DataFrame(
        [[customer[name] for name in SEGMENTATION_FEATURES]],
        columns=SEGMENTATION_FEATURES,
    )

    # Apply the scaler saved during preprocessing
    scaled_customer = segmentation_scaler.transform(
        customer_df
    )

    cluster = int(
        segmentation_model.predict(scaled_customer)[0]
    )

    return {
        "customer_cluster": cluster,
        "message": f"Customer assigned to cluster {cluster}",
    }
