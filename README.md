# AI-Powered Customer Experience Intelligence Platform

An end-to-end machine learning platform that analyzes customer sentiment, predicts customer churn, and identifies customer segments to help businesses understand customer behavior and improve customer experience.

## Project Overview

The **AI-Powered Customer Experience Intelligence Platform** combines deep learning and machine learning to analyze customer feedback, identify customers who may leave, and group customers according to their behavioral patterns.

The platform provides an interactive Streamlit dashboard connected to a FastAPI backend. Docker is used to containerize the frontend and backend for easier deployment.

## Key Features

- **Sentiment Analysis:** Uses an LSTM deep learning model to classify customer reviews as Positive, Negative, or Neutral.
- **Customer Churn Prediction:** Uses a Random Forest classifier to predict whether a customer is likely to churn.
- **Customer Segmentation:** Uses K-Means clustering to identify groups of customers based on purchasing behavior, satisfaction, and support interactions.
- **Interactive Dashboard:** Provides a user-friendly interface built with Streamlit.
- **REST API:** Serves model predictions through FastAPI endpoints.
- **Experiment Tracking:** Uses MLflow to track experiments and manage registered models.
- **Containerization:** Uses Docker Compose to run the frontend and backend as separate services.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **API Development:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit
- **Experiment Tracking:** MLflow
- **Model Persistence:** Pickle, Joblib, Keras model files
- **Deployment:** Docker, Docker Compose

## Project Architecture

```text
Customer Experience Intelligence Platform/
│
├── data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│   ├── sentiment_data.csv
│   └── external/
│
├── models/
│   ├── sentiment_model.keras
│   ├── sentiment_tokenizer.pkl
│   ├── sentiment_label_encoder.pkl
│   ├── churn_model.pkl
│   ├── churn_location_encoder.pkl
│   ├── churn_scaler.pkl
│   ├── churn_feature_names.pkl
│   ├── segmentation_model.pkl
│   ├── segmentation_scaler.pkl
│   └── segmentation_feature_names.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── train_churn.py
│   ├── train_segmentation.py
│   └── utils.py
│
├── backend/
│   ├── main.py
│   ├── schemas.py
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── register_sentiment.py
├── register_churn.py
├── register_segmentation.py
└── README.md
```

## Machine Learning Models

### 1. Sentiment Analysis

- **Model:** Long Short-Term Memory (LSTM)
- **Task:** Multiclass text classification
- **Input:** Customer review text
- **Output:** Positive, Negative, or Neutral sentiment
- **Evaluation:** Accuracy, precision, recall, and F1-score

The text preprocessing pipeline cleans customer reviews, converts text into sequences using a tokenizer, and pads the sequences before passing them to the LSTM model.

### 2. Customer Churn Prediction

- **Model:** Random Forest Classifier
- **Task:** Binary classification
- **Input:** Customer age, location, total orders, average order value, days since last order, support tickets, and customer satisfaction
- **Output:** Churn prediction and estimated churn probability
- **Evaluation:** Accuracy, precision, recall, and F1-score

The preprocessing pipeline scales numerical features and encodes customer location before generating predictions.

### 3. Customer Segmentation

- **Model:** K-Means Clustering
- **Task:** Unsupervised learning
- **Input:** Total orders, average order value, days since last order, support tickets, and customer satisfaction
- **Output:** Assigned customer cluster
- **Evaluation:** Silhouette score

The selected configuration uses six clusters. The observed silhouette score was approximately **0.1662**, indicating limited separation between clusters. Therefore, the segments should be interpreted as exploratory customer groups rather than perfectly distinct categories.

## Model Evaluation

The following results were observed during model development:

| Model | Metric | Observed Result |
|---|---|---:|
| Sentiment Analysis | Accuracy | Approximately 76–78% |
| Customer Churn | Accuracy | Approximately 71.5% |
| Customer Churn | Precision | Approximately 0.436 |
| Customer Churn | Recall | Approximately 0.494 |
| Customer Churn | F1-score | Approximately 0.463 |
| Customer Segmentation | Silhouette Score | Approximately 0.1662 |

These results are experimental results from the development process. Actual performance may vary depending on the data and evaluation setup.

## API Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict/sentiment` | Analyze customer sentiment |
| POST | `/predict/churn` | Predict customer churn |
| POST | `/predict/segment` | Assign a customer segment |

Interactive API documentation is available at:

`http://localhost:8000/docs`

## Dashboard

The Streamlit dashboard provides three main prediction interfaces:

1. **Sentiment Analysis:** Enter a customer review and view the predicted sentiment, confidence, and class probabilities.
2. **Churn Prediction:** Enter customer details and view the predicted churn status and estimated probability.
3. **Customer Segmentation:** Enter customer behavioral details and view the assigned cluster.

## MLflow Experiment Tracking

MLflow is used to track model experiments and manage registered models.

The registered models include:

- `Customer_Sentiment_Model`
- `Customer_Churn_Model`
- `Customer_Segmentation_Model`

The MLflow tracking interface was configured to use a local SQLite tracking database during development.

## Docker Deployment

The project uses two Docker containers:

- **Backend container:** Runs the FastAPI application on port 8000.
- **Frontend container:** Runs the Streamlit dashboard on port 8501.

### Prerequisites

- Docker Desktop
- Docker Compose
- Git (optional, for version control)

### Run the Project

Open a terminal in the project root directory.

Build the Docker images:

```bash
docker compose build
```

Start the application:

```bash
docker compose up -d
```

Check container status:

```bash
docker compose ps
```

Open the dashboard:

`http://localhost:8501`

Open the API documentation:

`http://localhost:8000/docs`

Stop the containers:

```bash
docker compose down
```