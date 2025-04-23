# 5. visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

def plot_clusters(X_scaled, labels, save_path=None):
    """
    Plot clusters using PCA-reduced 2D representation.

    Parameters:
        X_scaled (np.ndarray): Scaled feature array.
        labels (list or np.ndarray): Cluster labels.
        save_path (str): Optional path to save the plot. If None, displays the plot.
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_plot = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_plot['Cluster'] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_plot, palette='Set2', s=50)
    plt.title('Customer Segments (KMeans)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Plot saved to: {save_path}")
    else:
        plt.show()

