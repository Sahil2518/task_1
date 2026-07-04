import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def prepare_clustering_data(X, features, n_components=None):
    """
    Selects numerical features, scales them, and optionally applies PCA.
    
    Args:
        X (pd.DataFrame): Input dataframe.
        features (list): List of numerical column names to use.
        n_components (float or int, optional): Variance ratio or number of components for PCA.
                                               If None, PCA is skipped.
                                               
    Returns:
        X_prepared (np.ndarray): The transformed and scaled data ready for clustering.
        scaler (StandardScaler): The fitted scaler.
        pca (PCA or None): The fitted PCA model, or None if skipped.
    """
    X_selected = X[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    pca = None
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=42)
        X_prepared = pca.fit_transform(X_scaled)
    else:
        X_prepared = X_scaled
        
    return X_prepared, scaler, pca

def evaluate_clusters(X_prepared, k_range=range(2, 11), random_state=42):
    """
    Evaluates KMeans clustering over a range of k values.
    
    Returns:
        inertias (list): Sum of squared distances of samples to their closest cluster center.
        silhouette_scores (list): Mean silhouette coefficient for all samples.
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_prepared)
        
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_prepared, labels)
        silhouette_scores.append(score)
        
    return inertias, silhouette_scores

def plot_cluster_metrics(k_range, inertias, silhouette_scores, output_path):
    """
    Plots the Elbow Curve (inertia) and Silhouette Scores to help choose k.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Inertia (Elbow)
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia', color=color)
    ax1.plot(k_range, inertias, marker='o', color=color, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(k_range)
    ax1.grid(True, alpha=0.3)
    
    # Plot Silhouette
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)  
    ax2.plot(k_range, silhouette_scores, marker='s', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Cluster Evaluation: Elbow Method & Silhouette Score')
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
