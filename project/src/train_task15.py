"""
Task 15 - K-Means Clustering
PlaceMux Phase 1 Industry Immersion

Builds on the prepared clustering pipeline from Task 14 to:
  1. Run K-Means on the prepared (scaled + PCA) data.
  2. Evaluate quality with silhouette score and inertia.
  3. Profile each cluster's defining characteristics (pandas profiling).
  4. Name clusters in business terms.
  5. Check stability across 5 different random seeds.
  6. Recommend an action per segment.
  7. Visualise clusters with seaborn.
  8. Save all results for review.
"""

import os
import sys
import json
import joblib
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend – safe on any machine
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# ─── Constants ────────────────────────────────────────────────────────────────
SEED            = 42
PIPELINE_PATH   = "models/clustering_pipeline.pkl"
DATA_PATH       = "data/prepared_clusters.csv"
LOGS_DIR        = "logs"
PLOTS_DIR       = os.path.join(LOGS_DIR, "task15_plots")
RESULTS_JSON    = os.path.join(LOGS_DIR, "task15_cluster_analysis.json")
STABILITY_SEEDS = [0, 7, 13, 99, 2024]      # 5 seeds for stability check


# ─── Helper: load artefacts from Task 14 ─────────────────────────────────────
def load_pipeline_and_data():
    """Load the clustering pipeline and prepared dataset saved by Task 14."""
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Clustering pipeline not found at '{PIPELINE_PATH}'. "
            "Please run Task 14 first (run_task14.bat)."
        )
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Prepared cluster data not found at '{DATA_PATH}'. "
            "Please run Task 14 first (run_task14.bat)."
        )

    pipeline = joblib.load(PIPELINE_PATH)
    df_clustered = pd.read_csv(DATA_PATH)

    # Validate expected keys
    required_keys = {"features", "scaler", "pca", "kmeans", "best_k"}
    missing = required_keys - set(pipeline.keys())
    if missing:
        raise KeyError(
            f"Clustering pipeline is missing keys: {missing}. "
            "Re-run Task 14 to regenerate the pipeline."
        )

    return pipeline, df_clustered


# ─── Helper: re-transform raw features → PCA space ───────────────────────────
def get_prepared_features(df, pipeline):
    """
    Use the scaler and PCA from the pipeline to transform raw feature columns
    into the same space used to fit KMeans.
    """
    features = pipeline["features"]
    scaler   = pipeline["scaler"]
    pca      = pipeline["pca"]

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following expected feature columns are missing from the "
            f"data file: {missing_cols}"
        )

    X_scaled   = scaler.transform(df[features])
    X_prepared = pca.transform(X_scaled) if pca is not None else X_scaled
    return X_prepared


# ─── Step 1: Run K-Means ─────────────────────────────────────────────────────
def run_kmeans(X_prepared, best_k, seed=SEED):
    """Fit KMeans with the chosen k and return labels + fitted model."""
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_prepared)
    return km, labels


# ─── Step 2: Evaluate quality ────────────────────────────────────────────────
def evaluate_quality(X_prepared, labels, km):
    """Return silhouette score and inertia."""
    if len(set(labels)) < 2:
        raise ValueError("KMeans produced only 1 cluster – cannot compute silhouette score.")
    sil  = silhouette_score(X_prepared, labels)
    iner = km.inertia_
    return sil, iner


# ─── Step 3: Profile clusters ────────────────────────────────────────────────
def profile_clusters(df_raw_features, labels, features):
    """
    Compute per-cluster mean and std for every feature.
    Returns a tidy DataFrame with cluster profiles.
    """
    df_profile = df_raw_features[features].copy()
    df_profile["cluster"] = labels

    profile_mean = df_profile.groupby("cluster")[features].mean().round(3)
    profile_std  = df_profile.groupby("cluster")[features].std().round(3)
    profile_cnt  = df_profile.groupby("cluster").size().rename("count")

    return profile_mean, profile_std, profile_cnt


# ─── Step 4: Name clusters in business terms ─────────────────────────────────
def name_clusters(profile_mean: pd.DataFrame, features: list):
    """
    Automatically assigns a business name and recommended action
    based on key feature averages.

    Rules are driven by 'total_score', 'projects_completed',
    'active_days', and 'aptitude_score' – all present in Task 14 features.
    Each cluster is scored on multiple axes, then the highest-distinguishing
    axis determines the label.
    """
    naming = {}

    # Normalise profile to 0-1 range for easier comparison
    norm = (profile_mean - profile_mean.min()) / (
        profile_mean.max() - profile_mean.min() + 1e-9
    )

    score_col    = "total_score"       if "total_score"        in features else features[0]
    projects_col = "projects_completed" if "projects_completed" in features else None
    active_col   = "active_days"        if "active_days"        in features else None
    apt_col      = "aptitude_score"     if "aptitude_score"     in features else None

    for cluster_id in profile_mean.index:
        row = norm.loc[cluster_id]

        score_val    = row.get(score_col, 0.5)
        projects_val = row.get(projects_col, 0.5) if projects_col else 0.5
        active_val   = row.get(active_col, 0.5)   if active_col   else 0.5
        apt_val      = row.get(apt_col, 0.5)       if apt_col      else 0.5

        # Composite "engagement" = projects + active days
        engagement = (projects_val + active_val) / 2

        # Decision tree for business naming
        if score_val >= 0.6 and engagement >= 0.6:
            name   = "High-Performer (Engaged & Strong)"
            action = (
                "Fast-track to advanced placement opportunities. "
                "Offer leadership roles and premium job referrals."
            )
        elif score_val >= 0.6 and engagement < 0.4:
            name   = "Dormant Talent (High Score, Low Engagement)"
            action = (
                "Send personalised re-engagement emails highlighting "
                "active job listings. Schedule 1-on-1 career coaching calls."
            )
        elif score_val < 0.4 and engagement >= 0.6:
            name   = "Motivated but Under-Skilled"
            action = (
                "Enrol in targeted skill-gap courses for domain and aptitude. "
                "Pair with a mentor to convert effort into results."
            )
        elif apt_val >= 0.6 and projects_val < 0.4:
            name   = "Aptitude-Strong, Projects Lacking"
            action = (
                "Recommend hackathons and guided project templates. "
                "Gamify project completion with badges and leaderboard."
            )
        else:
            name   = "Early-Stage / Needs Activation"
            action = (
                "Trigger onboarding drip campaign with quick-win resources. "
                "A/B test push notifications to improve daily active usage."
            )

        naming[int(cluster_id)] = {"business_name": name, "recommended_action": action}

    return naming


# ─── Step 5: Stability check across seeds ────────────────────────────────────
def stability_check(X_prepared, best_k, reference_labels, seeds=STABILITY_SEEDS):
    """
    Fit KMeans on multiple seeds and compute ARI against the reference run.
    Returns mean and min ARI, plus per-seed scores.
    """
    ari_scores = {}
    for s in seeds:
        km_alt = KMeans(n_clusters=best_k, random_state=s, n_init=10)
        labels_alt = km_alt.fit_predict(X_prepared)
        ari = adjusted_rand_score(reference_labels, labels_alt)
        ari_scores[int(s)] = round(float(ari), 4)

    mean_ari = round(float(np.mean(list(ari_scores.values()))), 4)
    min_ari  = round(float(np.min(list(ari_scores.values()))), 4)
    return ari_scores, mean_ari, min_ari


# ─── Step 6: Visualisations ──────────────────────────────────────────────────
def save_cluster_heatmap(profile_mean, output_path):
    """Seaborn heatmap of normalised cluster feature means."""
    norm = (profile_mean - profile_mean.min()) / (
        profile_mean.max() - profile_mean.min() + 1e-9
    )
    fig, ax = plt.subplots(figsize=(max(10, len(profile_mean.columns) * 1.2), 5))
    sns.heatmap(
        norm,
        annot=profile_mean.values,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised Mean (0-1)"},
    )
    ax.set_title("Cluster Profile Heatmap – Normalised Feature Means", fontsize=13, pad=12)
    ax.set_xlabel("Features")
    ax.set_ylabel("Cluster")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap -> {output_path}")


def save_pca_scatter(X_prepared, labels, naming, output_path):
    """2-D PCA scatter plot coloured by cluster with business-name legend."""
    # If prepared data already has ≥2 components use first two; else reduce
    if X_prepared.shape[1] >= 2:
        coords = X_prepared[:, :2]
    else:
        pca2 = PCA(n_components=2, random_state=SEED)
        coords = pca2.fit_transform(X_prepared)

    df_plot = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels})
    palette = sns.color_palette("tab10", n_colors=len(set(labels)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (cid, grp) in enumerate(df_plot.groupby("cluster")):
        bname = naming.get(int(cid), {}).get("business_name", f"Cluster {cid}")
        ax.scatter(
            grp["PC1"], grp["PC2"],
            label=f"[{cid}] {bname}",
            color=palette[i],
            alpha=0.55,
            s=30,
            edgecolors="none",
        )
    ax.set_title("Cluster Distribution (PCA Space)", fontsize=13)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc="best", fontsize=8, title="Clusters")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot -> {output_path}")


def save_bar_profiles(profile_mean, naming, output_path):
    """Per-cluster bar chart showing raw feature means."""
    n_clusters = len(profile_mean)
    fig, axes  = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5), sharey=False)
    if n_clusters == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=n_clusters)
    for ax, (cid, row) in zip(axes, profile_mean.iterrows()):
        bname = naming.get(int(cid), {}).get("business_name", f"Cluster {cid}")
        ax.barh(row.index, row.values, color=palette[cid], edgecolor="white")
        ax.set_title(f"Cluster {cid}\n{bname}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean Value")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Cluster Feature Profiles", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved bar profiles -> {output_path}")


def save_stability_chart(ari_scores, mean_ari, output_path):
    """Bar chart of ARI scores across stability seeds."""
    seeds  = list(ari_scores.keys())
    scores = list(ari_scores.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#2ecc71" if s >= 0.8 else "#e67e22" if s >= 0.5 else "#e74c3c" for s in scores]
    ax.bar([str(s) for s in seeds], scores, color=colors, edgecolor="white")
    ax.axhline(mean_ari, color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean ARI = {mean_ari:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Cluster Stability Across Random Seeds", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved stability chart -> {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Task 15 -- K-Means Clustering")
    print("  PlaceMux Phase 1 Industry Immersion")
    print("=" * 65)

    # ── 0. Load Task-14 artefacts ────────────────────────────────────────────
    print("\n[0/7] Loading Task-14 clustering pipeline and prepared data...")
    try:
        pipeline, df_clustered = load_pipeline_and_data()
    except (FileNotFoundError, KeyError) as exc:
        print(f"\n  [ERROR] {exc}")
        print("  Aborting. Please run Task 14 before Task 15.\n")
        sys.exit(1)

    best_k   = pipeline["best_k"]
    features = pipeline["features"]
    print(f"  best_k from Task 14 : {best_k}")
    print(f"  Features            : {features}")
    print(f"  Dataset shape       : {df_clustered.shape}")

    # ── 1. Run K-Means ───────────────────────────────────────────────────────
    print(f"\n[1/7] Running K-Means (k={best_k}, seed={SEED}) on prepared data...")
    try:
        X_prepared = get_prepared_features(df_clustered, pipeline)
        km, labels = run_kmeans(X_prepared, best_k, seed=SEED)
        print(f"  Cluster sizes : { {i: int((labels==i).sum()) for i in range(best_k)} }")
    except Exception as exc:
        print(f"\n  [ERROR] Failed during K-Means fitting: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── 2. Evaluate quality ─────────────────────────────────────────────────
    print("\n[2/7] Evaluating cluster quality...")
    try:
        sil, iner = evaluate_quality(X_prepared, labels, km)
        print(f"  Silhouette Score : {sil:.4f}  (higher is better; >0.5 = strong)")
        print(f"  Inertia          : {iner:.2f}")
    except Exception as exc:
        print(f"\n  [WARNING] Quality evaluation failed: {exc}")
        sil, iner = None, None

    # ── 3. Profile clusters ─────────────────────────────────────────────────
    print("\n[3/7] Profiling clusters (pandas groupby)...")
    try:
        profile_mean, profile_std, profile_cnt = profile_clusters(
            df_clustered, labels, features
        )
        print("\n  --- Cluster Mean Profiles ---")
        print(profile_mean.to_string())
        print("\n  --- Cluster Sizes ---")
        print(profile_cnt.to_string())
    except Exception as exc:
        print(f"\n  [ERROR] Cluster profiling failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ── 4. Name clusters ────────────────────────────────────────────────────
    print("\n[4/7] Naming clusters in business terms...")
    try:
        naming = name_clusters(profile_mean, features)
        for cid, info in naming.items():
            print(f"\n  Cluster {cid} — \"{info['business_name']}\"")
            print(f"    Recommended action: {info['recommended_action']}")
    except Exception as exc:
        print(f"\n  [WARNING] Auto-naming failed: {exc}. Falling back to generic names.")
        naming = {i: {"business_name": f"Segment {i}", "recommended_action": "Investigate further."} for i in range(best_k)}

    # ── 5. Stability check ──────────────────────────────────────────────────
    print("\n[5/7] Checking cluster stability across seeds...")
    try:
        ari_scores, mean_ari, min_ari = stability_check(X_prepared, best_k, labels)
        print(f"  ARI per seed : {ari_scores}")
        print(f"  Mean ARI     : {mean_ari:.4f}  (>0.80 = stable, 0.50-0.80 = moderate)")
        print(f"  Min  ARI     : {min_ari:.4f}")
        stability_verdict = (
            "STABLE"   if mean_ari >= 0.80 else
            "MODERATE" if mean_ari >= 0.50 else
            "UNSTABLE"
        )
        print(f"  Verdict      : {stability_verdict}")
    except Exception as exc:
        print(f"\n  [WARNING] Stability check failed: {exc}")
        ari_scores, mean_ari, min_ari = {}, None, None
        stability_verdict = "UNKNOWN"

    # ── 6. Visualisations ───────────────────────────────────────────────────
    print("\n[6/7] Generating visualisations (seaborn)...")
    try:
        save_cluster_heatmap(
            profile_mean,
            os.path.join(PLOTS_DIR, "cluster_heatmap.png")
        )
        save_pca_scatter(
            X_prepared, labels, naming,
            os.path.join(PLOTS_DIR, "cluster_pca_scatter.png")
        )
        save_bar_profiles(
            profile_mean, naming,
            os.path.join(PLOTS_DIR, "cluster_bar_profiles.png")
        )
        save_stability_chart(
            ari_scores, mean_ari,
            os.path.join(PLOTS_DIR, "cluster_stability.png")
        )
    except Exception as exc:
        print(f"\n  [WARNING] One or more plots failed: {exc}")
        traceback.print_exc()

    # ── 7. Save results ─────────────────────────────────────────────────────
    print("\n[7/7] Saving analysis results...")
    try:
        results = {
            "task":             "Task 15 — K-Means Clustering",
            "seed":             SEED,
            "best_k":           int(best_k),
            "silhouette_score": float(sil) if sil is not None else None,
            "inertia":          float(iner) if iner is not None else None,
            "cluster_sizes":    { str(i): int((labels == i).sum()) for i in range(best_k) },
            "cluster_profiles": {
                str(cid): {
                    "mean":             profile_mean.loc[cid].to_dict(),
                    "std":              profile_std.loc[cid].to_dict(),
                    "count":            int(profile_cnt.loc[cid]),
                    "business_name":    naming[cid]["business_name"],
                    "recommended_action": naming[cid]["recommended_action"],
                }
                for cid in profile_mean.index
            },
            "stability": {
                "seeds_tested":   STABILITY_SEEDS,
                "ari_per_seed":   ari_scores,
                "mean_ari":       mean_ari,
                "min_ari":        min_ari,
                "verdict":        stability_verdict,
            },
        }

        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Results saved -> {RESULTS_JSON}")
    except Exception as exc:
        print(f"\n  [ERROR] Failed to save results JSON: {exc}")
        traceback.print_exc()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TASK 15 -- COMPLETE")
    print("=" * 65)
    print(f"  k                : {best_k}")
    if sil is not None:
        print(f"  Silhouette Score : {sil:.4f}")
    print(f"  Stability Verdict: {stability_verdict}")
    print(f"  Plots saved to   : {PLOTS_DIR}/")
    print(f"  Results JSON     : {RESULTS_JSON}")
    print("\n  Segment Summary:")
    for cid, info in naming.items():
        cnt = int((labels == cid).sum())
        print(f"    [{cid}] {info['business_name']} ({cnt} users)")
    print()


if __name__ == "__main__":
    main()
