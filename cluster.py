# 3. cluster.py

from sklearn.cluster import KMeans
import joblib

def apply_kmeans(X_scaled, n_clusters=4, model_path="kmeans_model.pkl"):
    # Train the model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Optional: show inertia score
    print(f"KMeans model trained with inertia: {kmeans.inertia_:.2f}")

    # Save the model
    joblib.dump(kmeans, model_path)
    print(f"Model saved to: {model_path}")

    return kmeans
