
import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore


# Configuration

DATA_PATH = "data/processed_data.csv"
MAX_WORDS = 5000
MAX_LENGTH = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Run this script from the project root folder."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CUSTOMER EXPERIENCE INTELLIGENCE PLATFORM")
print("DATA PREPROCESSING")
print("=" * 60)

# Verify that the dataset contains the required columns.
required_columns = [
    "review_text",
    "sentiment",
    "age",
    "location",
    "total_orders",
    "avg_order_value",
    "days_since_last_order",
    "support_tickets",
    "customer_satisfaction",
    "churn"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

# Check for missing values in columns used by this project
if df[required_columns].isnull().any().any():
    missing_counts = df[required_columns].isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    raise ValueError(
        "Missing values found in required columns:\n"
        f"{missing_counts}\n"
        "Handle these missing values before preprocessing."
    )

if df.empty:
    raise ValueError("The dataset is empty.")

print(f"\nDataset loaded successfully: {DATA_PATH}")
print("Dataset shape:", df.shape)

# PART 1: SENTIMENT ANALYSIS PREPROCESSING

SENTIMENT_DATA_PATH = "data/sentiment_data.csv"


def clean_text(text):
    #Normalize review text for sentiment analysis
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load the new sentiment dataset separately
if not os.path.exists(SENTIMENT_DATA_PATH):
    raise FileNotFoundError(
        f"Sentiment dataset not found: {SENTIMENT_DATA_PATH}"
    )

sentiment_df = pd.read_csv(SENTIMENT_DATA_PATH)

required_sentiment_columns = ["review_text", "sentiment"]

if not all(
    column in sentiment_df.columns
    for column in required_sentiment_columns
):
    raise ValueError(
        "The sentiment dataset must contain review_text and sentiment."
    )

sentiment_df = sentiment_df[required_sentiment_columns].copy()

# Remove missing and empty reviews
sentiment_df = sentiment_df.dropna(
    subset=["review_text", "sentiment"]
)

sentiment_df["review_text"] = (
    sentiment_df["review_text"].astype(str).str.strip()
)

sentiment_df["sentiment"] = (
    sentiment_df["sentiment"].astype(str).str.strip().str.title()
)

sentiment_df = sentiment_df[
    (sentiment_df["review_text"] != "")
    & (sentiment_df["sentiment"] != "")
].copy()

# Keep only the three expected sentiment classes
valid_classes = ["Negative", "Neutral", "Positive"]

sentiment_df = sentiment_df[
    sentiment_df["sentiment"].isin(valid_classes)
].copy()

sentiment_df["cleaned_review"] = (
    sentiment_df["review_text"].apply(clean_text)
)

sentiment_df = sentiment_df[
    sentiment_df["cleaned_review"] != ""
].copy()

# Find cleaned review texts that have conflicting labels
label_counts = (
    sentiment_df.groupby("cleaned_review")["sentiment"].nunique()
)

conflicting_reviews = label_counts[label_counts > 1]

print("Sentiment dataset loaded:", SENTIMENT_DATA_PATH)
print("Rows before conflict removal:", len(sentiment_df))
print("Conflicting cleaned review texts:", len(conflicting_reviews))

# Remove all versions of ambiguous reviews rather than guessing which sentiment label is correct
if not conflicting_reviews.empty:
    sentiment_df = sentiment_df[
        ~sentiment_df["cleaned_review"].isin(
            conflicting_reviews.index
        )
    ].copy()

# Keep one example per cleaned review text
sentiment_df = sentiment_df.drop_duplicates(
    subset="cleaned_review"
).reset_index(drop=True)

if sentiment_df.empty:
    raise ValueError("No usable sentiment reviews remain.")

if set(sentiment_df["sentiment"].unique()) != set(valid_classes):
    raise ValueError(
        "The dataset must contain Negative, Neutral, and Positive "
        "reviews after cleaning."
    )

print("Unique usable review texts:", len(sentiment_df))
print("Sentiment distribution:")
print(sentiment_df["sentiment"].value_counts())

# Split unique reviews into training and testing sets
X_train_text, X_test_text, y_train_text, y_test_text = (
    train_test_split(
        sentiment_df["cleaned_review"],
        sentiment_df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=sentiment_df["sentiment"]
    )
)

# Verify there is no text overlap
overlap = set(X_train_text) & set(X_test_text)

if overlap:
    raise RuntimeError(
        "Data leakage detected: review texts overlap between "
        "training and testing sets."
    )

print("Sentiment train/test split completed.")
print("Training rows:", len(X_train_text))
print("Testing rows:", len(X_test_text))
print("Training unique reviews:", X_train_text.nunique())
print("Testing unique reviews:", X_test_text.nunique())
print("Review-text overlap:", len(overlap))

print("Training sentiment distribution:")
print(y_train_text.value_counts())

print("Testing sentiment distribution:")
print(y_test_text.value_counts())

# Fit the tokenizer only on training reviews.
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train_text)

X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
X_test_sequences = tokenizer.texts_to_sequences(X_test_text)

X_train_sentiment = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_sentiment = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# Encode sentiment labels.
label_encoder = LabelEncoder()

y_train_sentiment = label_encoder.fit_transform(y_train_text)
y_test_sentiment = label_encoder.transform(y_test_text)

print("Encoded sentiment classes:")
print(label_encoder.classes_)

print("X_train_sentiment shape:", X_train_sentiment.shape)
print("X_test_sentiment shape:", X_test_sentiment.shape)
print("y_train_sentiment shape:", y_train_sentiment.shape)
print("y_test_sentiment shape:", y_test_sentiment.shape)

# Save sentiment arrays and preprocessing objects.
# These filenames are already used by src/train.py.
np.save("data/X_train_sentiment.npy", X_train_sentiment)
np.save("data/X_test_sentiment.npy", X_test_sentiment)
np.save("data/y_train_sentiment.npy", y_train_sentiment)
np.save("data/y_test_sentiment.npy", y_test_sentiment)

joblib.dump(tokenizer, "models/sentiment_tokenizer.pkl")
joblib.dump(label_encoder, "models/sentiment_label_encoder.pkl")

print("Sentiment preprocessing completed successfully.")


# PART 2: CHURN PREDICTION PREPROCESSING

churn_features = [
    "age",
    "location",
    "total_orders",
    "avg_order_value",
    "days_since_last_order",
    "support_tickets",
    "customer_satisfaction"
]

numerical_features = [
    "age",
    "total_orders",
    "avg_order_value",
    "days_since_last_order",
    "support_tickets",
    "customer_satisfaction"
]

categorical_features = ["location"]

X_churn = df[churn_features].copy()
y_churn = df["churn"].copy()

if y_churn.nunique() < 2:
    raise ValueError(
        "Churn prediction requires at least two target classes."
    )

# Split before fitting encoders or scalers to avoid data leakage.
X_churn_train, X_churn_test, y_churn_train, y_churn_test = (train_test_split(X_churn, y_churn, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_churn))

print("Churn target distribution:")
print(y_churn.value_counts())

print("Churn training shape:", X_churn_train.shape)
print("Churn testing shape:", X_churn_test.shape)

# One-hot encode locations.
one_hot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

location_train_encoded = one_hot_encoder.fit_transform(
    X_churn_train[categorical_features]
)

location_test_encoded = one_hot_encoder.transform(
    X_churn_test[categorical_features]
)

location_columns = one_hot_encoder.get_feature_names_out(
    categorical_features
)

location_train_df = pd.DataFrame(
    location_train_encoded,
    columns=location_columns,
    index=X_churn_train.index
)

location_test_df = pd.DataFrame(
    location_test_encoded,
    columns=location_columns,
    index=X_churn_test.index
)

# Scale numerical features using training data only

scaler = StandardScaler()

X_churn_train_scaled = scaler.fit_transform(
    X_churn_train[numerical_features]
)

X_churn_test_scaled = scaler.transform(
    X_churn_test[numerical_features]
)

X_churn_train_scaled_df = pd.DataFrame(
    X_churn_train_scaled,
    columns=numerical_features,
    index=X_churn_train.index
)

X_churn_test_scaled_df = pd.DataFrame(
    X_churn_test_scaled,
    columns=numerical_features,
    index=X_churn_test.index
)

# Combine numerical and categorical features.
X_churn_train_processed = pd.concat(
    [X_churn_train_scaled_df, location_train_df],
    axis=1
)

X_churn_test_processed = pd.concat(
    [X_churn_test_scaled_df, location_test_df],
    axis=1
)

# Preserve feature order for future inference
churn_feature_names = list(X_churn_train_processed.columns)

if list(X_churn_test_processed.columns) != churn_feature_names:
    raise RuntimeError(
        "Training and testing churn feature columns do not match."
    )

# Save processed churn datasets
np.save(
    "data/X_train_churn.npy",
    X_churn_train_processed.to_numpy()
)

np.save(
    "data/X_test_churn.npy",
    X_churn_test_processed.to_numpy()
)

np.save("data/y_train_churn.npy", y_churn_train.to_numpy())
np.save("data/y_test_churn.npy", y_churn_test.to_numpy())

joblib.dump(
    one_hot_encoder,
    "models/churn_location_encoder.pkl"
)

joblib.dump(scaler, "models/churn_scaler.pkl")

joblib.dump(
    churn_feature_names,
    "models/churn_feature_names.pkl"
)

print("Processed churn training shape:",
      X_churn_train_processed.shape)

print("Processed churn testing shape:",
      X_churn_test_processed.shape)

print("Churn preprocessing completed successfully.")


# PART 3: CUSTOMER SEGMENTATION PREPROCESSING

segmentation_features = [
    "total_orders",
    "avg_order_value",
    "days_since_last_order",
    "support_tickets",
    "customer_satisfaction"
]

segmentation_df = df[segmentation_features].copy()

# Segmentation currently uses the full customer dataset
# Fit one scaler and save it for consistent future predictions

segmentation_scaler = StandardScaler()

X_segmentation = segmentation_scaler.fit_transform(
    segmentation_df
)

np.save("data/X_segmentation.npy", X_segmentation)

joblib.dump(
    segmentation_scaler,
    "models/segmentation_scaler.pkl"
)

joblib.dump(
    segmentation_features,
    "models/segmentation_feature_names.pkl"
)

print("Segmentation input shape:", segmentation_df.shape)
print("Scaled segmentation shape:", X_segmentation.shape)

print("Customer segmentation preprocessing completed successfully.")
