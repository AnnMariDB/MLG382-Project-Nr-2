# 6. app.py

from load_data import load_data
from preprocess import preprocess_data
from cluster import apply_kmeans
from visualize import plot_clusters

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        X_scaled, df_clean, scaler = preprocess_data(df)
        model = apply_kmeans(X_scaled)
        plot_clusters(X_scaled, model.labels_)
    else:
        print("[ERROR] Data loading failed. Exiting.")


