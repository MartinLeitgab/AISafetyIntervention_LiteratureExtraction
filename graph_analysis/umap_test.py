import json
import pandas as pd
from pathlib import Path

# Load all UMAP test results
results = []

for json_file in Path(".").glob("clusters_*UMAP*.json"):
    with open(json_file) as f:
        data = json.load(f)

    # Extract config info from filename
    parts = json_file.stem.replace("clusters_", "").split("_")
    umap_dim = int(parts[-1].replace("UMAP", ""))

    # Extract metrics per algorithm
    for algo, algo_data in data["results"].items():
        if "metrics" in algo_data:
            results.append(
                {
                    "edge_config": data["edge_config"],
                    "mode": data["mode"],
                    "node_type": data["node_type"],
                    "umap_dim": umap_dim,
                    "algorithm": algo,
                    "silhouette": algo_data["metrics"].get("silhouette", -1),
                    "n_clusters": algo_data["n_clusters"],
                    "edge_validation": algo_data["metrics"].get(
                        "edge_validation_overall", 0
                    ),
                }
            )

df = pd.DataFrame(results)

# Summarize by dimension
print("\n=== Silhouette Scores by UMAP Dimension ===")
summary = (
    df.groupby("umap_dim")
    .agg({"silhouette": ["mean", "median", "min", "max"], "n_clusters": "mean"})
    .round(3)
)
print(summary)

# Best per config
print("\n=== Best Dimension per Config ===")
best = df.loc[
    df.groupby(["edge_config", "mode", "node_type", "algorithm"])["silhouette"].idxmax()
]
print(best.groupby("umap_dim").size().sort_values(ascending=False))

# Save full results
df.to_csv("umap_dimension_scan_results.csv", index=False)
print("\n✓ Saved: umap_dimension_scan_results.csv")
