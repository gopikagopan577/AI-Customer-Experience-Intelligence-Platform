
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import load_numpy_data


# PART 1: LOAD SEGMENTATION DATA

X = load_numpy_data("data/X_segmentation.npy")

print("Segmentation dataset loaded successfully!")
print("Dataset shape:", X.shape)

if X.ndim != 2 or X.shape[0] < 3:
    raise ValueError("Segmentation data must have at least 3 rows and features.")

if not np.isfinite(X).all():
    raise ValueError("Segmentation data contains NaN or infinite values.")


# PART 2: FIND A SUITABLE NUMBER OF CLUSTERS

MAX_K = min(8, X.shape[0] - 1)
k_values = list(range(2, MAX_K + 1))

inertias = []
silhouette_scores = []


for k in k_values:
    candidate_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = candidate_model.fit_predict(X)

    inertias.append(candidate_model.inertia_)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

    print(f"K={k}: Inertia={candidate_model.inertia_:.2f}, "
          f"Silhouette Score={score:.4f}")


# Choose the K with the highest silhouette score
best_index = int(np.argmax(silhouette_scores))
BEST_K = k_values[best_index]

print(f"\nSelected number of clusters: {BEST_K}")
print(f"Best silhouette score: {silhouette_scores[best_index]:.4f}")


# PART 3: VISUALIZE CLUSTER EVALUATION

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_values, inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)

axes[1].plot(k_values, silhouette_scores, marker="o")
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

fig.tight_layout()
plt.show()
plt.close(fig)


# PART 4: TRAIN AND TRACK FINAL K-MEANS MODEL

mlflow.set_experiment("Customer_Segmentation")

segmentation_model = KMeans(
    n_clusters=BEST_K,
    random_state=42,
    n_init=10
)

with mlflow.start_run(run_name="Customer_KMeans_v1"):

    segmentation_labels = segmentation_model.fit_predict(X)

    final_silhouette = silhouette_score(X, segmentation_labels)

    mlflow.log_params({
        "model_type": "KMeans",
        "n_clusters": BEST_K,
        "random_state": 42,
        "n_init": 10
    })

    mlflow.log_metrics({
        "inertia": float(segmentation_model.inertia_),
        "silhouette_score": float(final_silhouette)
    })

    mlflow.sklearn.log_model(
        segmentation_model,
        name="customer_kmeans_model",
        serialization_format="pickle"
    )

    print("Final Segmentation Results")
    print("Number of clusters:", BEST_K)
    print(f"Inertia: {segmentation_model.inertia_:.2f}")
    print(f"Silhouette score: {final_silhouette:.4f}")

    unique_clusters, cluster_counts = np.unique(
        segmentation_labels,
        return_counts=True
    )

    print("Customers per cluster:")
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        print(f"Cluster {cluster_id}: {count} customers")



# PART 5: SAVE MODEL AND CUSTOMER SEGMENTS

os.makedirs("models", exist_ok=True)

joblib.dump(
    segmentation_model,
    "models/segmentation_model.pkl"
)

np.save(
    "data/customer_segments.npy",
    segmentation_labels
)

print("Segmentation model saved:")
print("models/segmentation_model.pkl")
print("Customer cluster assignments saved:")
print("data/customer_segments.npy")
print("Customer segmentation training completed!")
