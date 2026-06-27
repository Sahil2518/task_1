import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
from src.data import generate_placemux_data, feature_engineering

def plot_importances(model, feature_names, title, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    os.makedirs("models", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"models/{filename}")
    print(f"Saved feature importance plot to models/{filename}")

def main():
    print("--- PlaceMux Baseline Target Feature Engineering ---")
    
    # 1. Generate data with ALL features (including leaky ones)
    print("\n1. Generating raw PlaceMux data (including time-based and leaky features)...")
    df_raw = generate_placemux_data(n_samples=1500, seed=42)
    
    # 2. Derive features without pruning
    print("2. Deriving domain & aggregate features (keeping leaks for demonstration)...")
    df_unpruned = feature_engineering(df_raw, prune_leaks=False)
    
    # Handle NaNs for the dummy model (days_to_placement has NaNs)
    df_unpruned['days_to_placement'] = df_unpruned['days_to_placement'].fillna(-1)
    
    # Drop datetime columns as models can't handle them directly
    df_unpruned = df_unpruned.drop(columns=['registration_date', 'last_login_date'])
    
    X_unpruned = df_unpruned.drop(columns=['placed'])
    y = df_unpruned['placed']
    
    # 3. Train a model to inspect feature importance
    print("3. Training a RandomForest to inspect feature importance (Leakage Check)...")
    model_leaky = RandomForestClassifier(n_estimators=100, random_state=42)
    model_leaky.fit(X_unpruned, y)
    
    plot_importances(model_leaky, X_unpruned.columns, "Feature Importances (With Leaky Features)", "importances_leaky.png")
    
    print("\n[LEAKAGE DETECTED] Notice how 'has_offer_letter' and 'days_to_placement' dominate the importance.")
    print("These are leaky because they are direct consequences of the target (placed).")
    
    # 4. Prune useless/leaky features
    print("\n4. Pruning useless and leaky features...")
    df_pruned = feature_engineering(df_raw, prune_leaks=True)
    X_pruned = df_pruned.drop(columns=['placed'])
    
    print("5. Retraining model on locked baseline feature set...")
    model_locked = RandomForestClassifier(n_estimators=100, random_state=42)
    model_locked.fit(X_pruned, y)
    
    plot_importances(model_locked, X_pruned.columns, "Baseline Feature Importances (Locked)", "importances_locked.png")
    
    print("\n[BASELINE LOCKED] Features used in the final pipeline:")
    for col in X_pruned.columns:
        print(f" - {col}")
        
    print("\nTask 7 Execution Complete. The clean data pipeline in `src/data.py` is now locked for the project.")

if __name__ == "__main__":
    main()
