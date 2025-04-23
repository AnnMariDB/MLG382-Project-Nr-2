# 3. cluster.py

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def train_kmeans(X, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return model, labels, score

def train_dbscan(X, eps=0.9, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return model, labels, score

