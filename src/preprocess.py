import pandas as pd
import re
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

df = pd.read_csv("data/processed_data.csv")


# Select data needed for sentiment analysis
sentiment_df = df[["review_text", "sentiment"]].copy()

print(sentiment_df.head())
print(sentiment_df.shape)

# To clean review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

sentiment_df["cleaned_review"] = sentiment_df["review_text"].apply(clean_text)

print(sentiment_df[["review_text", "cleaned_review"]].head())

# X,y split
X = sentiment_df["cleaned_review"]
y = sentiment_df["sentiment"]

print('X shape:', X.shape)
print('y shape:', y.shape)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape)
print(X_test.shape)

# Create tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")


# Learn words from training data
tokenizer.fit_on_texts(X_train)


# Convert text into numbers
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


print("First training sequence:")
print(X_train[0])

#padding
# Pad sequences to the same length
max_length = 100

X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")

X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")


# Check padded data
print(X_train.shape)
print(X_test.shape)

print("First padded training sequence:")
print(X_train[0])

#label encoding
# Encode sentiment labels
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print("Sentiment classes:")
print(label_encoder.classes_)

print("Encoded training labels:")
print(y_train[:10])

print("Encoded testing labels:")
print(y_test[:10])

# Create folders if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Save processed sentiment data
np.save("data/X_train_sentiment.npy", X_train)
np.save("data/X_test_sentiment.npy", X_test)
np.save("data/y_train_sentiment.npy", y_train)
np.save("data/y_test_sentiment.npy", y_test)


# Save tokenizer
joblib.dump(tokenizer, "models/sentiment_tokenizer.pkl")

# Save label encoder
joblib.dump(label_encoder, "models/sentiment_label_encoder.pkl")


print("Sentiment preprocessing files saved successfully.")

# Churn Preprocessing

# Select data needed for churn prediction
churn_features = ["age", "location", "total_orders", "avg_order_value", "days_since_last_order", "support_tickets", "customer_satisfaction"]

churn_df = df[churn_features + ["churn"]].copy()

print("Churn data:")
print(churn_df.head())

print("Churn data shape:", churn_df.shape)

print("Churn target distribution:")
print(churn_df["churn"].value_counts())

# Train/Test Split for Churn
# Separate churn features and target
X_churn = churn_df[churn_features]
y_churn = churn_df["churn"]

print("Churn X shape:", X_churn.shape)
print("Churn y shape:", y_churn.shape)


# Split churn data into training and testing sets
X_churn_train, X_churn_test, y_churn_train, y_churn_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn)

print("Churn training data:", X_churn_train.shape)
print("Churn testing data:", X_churn_test.shape)

print("Churn training target:", y_churn_train.shape)
print("Churn testing target:", y_churn_test.shape)