"""
Task 14 - Data Cluster Parameter Prep

This script prepares data for unsupervised clustering by:
  1. Loading the processed dataset.
  2. Selecting numeric features to define segments.
  3. Scaling features (StandardScaler) to ensure no single feature dominates.
  4. Optionally reducing dimensions (PCA).
  5. Iterating through candidate k values and evaluating with Silhouette Score & Inertia.
  6. Plotting the results to justify the choice of k.
  7. Locking the prepared dataset and parameters for future use.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data import load_data
from src.cluster import prepare_clustering_data, evaluate_clusters, plot_cluster_metrics
from sklearn.cluster import KMeans

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("\n" + "=" * 65)
    print("  Task 14 -- Data Cluster Parameter Prep")
    print("  PlaceMux Phase 1 Industry Immersion")
    print("=" * 65)

    # 1. Load data
    print("\n[1/6] Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # For unsupervised learning, we often use all available data to find segments.
    # However, to be strictly rigorous, we can fit our Scaler/PCA on train and transform all.
    # We will combine all X data to output a fully preprocessed dataset.
    X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)
    
    # 2. Select Features
    print("\n[2/6] Selecting numeric features for clustering...")
    numeric_features = [
        "domain_score", 
        "aptitude_score", 
        "projects_completed", 
        "active_days", 
        "registration_month", 
        "total_score"
    ]
    print(f"  Selected features: {numeric_features}")

    # 3 & 4. Scale and Reduce Dimensions
    print("\n[3/6] Scaling features (StandardScaler) and applying PCA (95% variance)...")
    # We fit on X_train to prevent any potential data leakage (strict baseline mindset)
    X_train_prepared, scaler, pca = prepare_clustering_data(
        X_train, 
        features=numeric_features, 
        n_components=0.95
    )
    
    # Transform the full dataset
    X_all_scaled = scaler.transform(X_all[numeric_features])
    X_all_prepared = pca.transform(X_all_scaled) if pca else X_all_scaled
    
    print(f"  Original features: {len(numeric_features)}")
    print(f"  Features after PCA: {X_all_prepared.shape[1]}")

    # 5. Choose candidate k
    print("\n[4/6] Evaluating clusters (k=2 to 10) on training set...")
    k_range = range(2, 11)
    inertias, silhouette_scores = evaluate_clusters(X_train_prepared, k_range=k_range)
    
    # 6. Plot and select k
    plot_path = "logs/cluster_evaluation.png"
    plot_cluster_metrics(k_range, inertias, silhouette_scores, plot_path)
    print(f"  Saved evaluation plot -> {plot_path}")
    
    # Automatically select k with the highest silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n  -> Justified Choice of k: {best_k} (Max Silhouette Score: {max(silhouette_scores):.4f})")
    
    # Sanity check distances (fit KMeans with best k on full data)
    print(f"\n[5/6] Fitting KMeans (k={best_k}) on full prepared dataset...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    full_labels = kmeans.fit_predict(X_all_prepared)
    
    # Add labels to the original feature set for inspection
    df_clustered = X_all[numeric_features].copy()
    df_clustered['cluster'] = full_labels
    
    # 7. Lock prepared dataset and parameters
    print("\n[6/6] Locking parameters and prepared dataset...")
    df_clustered.to_csv("data/prepared_clusters.csv", index=False)
    print("  Prepared dataset saved -> data/prepared_clusters.csv")
    
    clustering_pipeline = {
        "features": numeric_features,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "best_k": best_k
    }
    joblib.dump(clustering_pipeline, "models/clustering_pipeline.pkl")
    print("  Clustering pipeline saved -> models/clustering_pipeline.pkl")
    
    # Save parameters to JSON
    params = {
        "features_used": numeric_features,
        "pca_components": int(pca.n_components_) if pca else len(numeric_features),
        "best_k": int(best_k),
        "max_silhouette_score": float(max(silhouette_scores)),
        "cluster_centers_shape": kmeans.cluster_centers_.shape
    }
    with open("logs/task14_clustering_params.json", "w") as f:
        json.dump(params, f, indent=4)
        
    print("\n" + "=" * 65)
    print("  TASK 14 -- COMPLETE")
    print("=" * 65)
    print("  Ready for segment-specific analysis or serving.")

if __name__ == "__main__":
    main()
