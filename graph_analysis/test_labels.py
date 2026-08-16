"""
Debug category composition per path
"""

import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Load cache
with open("pathway_cache_final.pkl", "rb") as f:
    cache = pickle.load(f)

comp_c = cache["comp_c"]

print(f"Total paths: {len(comp_c)}")
print("\nFirst 5 path compositions:")
for i, c in enumerate(comp_c[:5]):
    print(
        f"  Path {i + 1}: length={c['length']}, cats={c['categories']}, n_unique={c['n_unique_cats']}"
    )

# Analyze categories
all_cats = Counter()
for c in comp_c:
    for cat, cnt in c["categories"].items():
        all_cats[cat] += cnt

print("\nTotal category counts across all paths:")
for cat, cnt in all_cats.most_common():
    print(f"  {cat}: {cnt}")

# Build heatmap data
CAT_ORDER = [
    "risk",
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]

bins = list(range(1, 21)) + [">20"]
mat = np.zeros((len(bins), len(CAT_ORDER)))
counts = [0] * len(bins)

for c in comp_c:
    length = c["length"]
    idx = min(length - 1, len(bins) - 1)
    counts[idx] += 1

    for cat, cnt in c["categories"].items():
        if cat in CAT_ORDER:
            mat[idx, CAT_ORDER.index(cat)] += cnt

print("\nPaths per length:")
for i, (bin_label, count) in enumerate(zip(bins, counts)):
    if count > 0:
        print(f"  {bin_label} hops: {count} paths")

# Average
for i, cnt in enumerate(counts):
    if cnt > 0:
        mat[i, :] /= cnt

print("\nHeatmap matrix (first 10 rows):")
print(f"Columns: {CAT_ORDER}")
for i in range(min(10, len(bins))):
    if counts[i] > 0:
        print(f"{bins[i]:>3}: {mat[i, :]}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
ax.set_xticks(np.arange(len(CAT_ORDER)))
ax.set_xticklabels([c.replace(" ", "\n") for c in CAT_ORDER], fontsize=9)
ax.set_yticks(np.arange(len(bins)))
ax.set_yticklabels(bins, fontsize=9)
ax.set_xlabel("Category", fontsize=11, fontweight="bold")
ax.set_ylabel("Hops", fontsize=11, fontweight="bold")
ax.set_title("Avg Category Frequency per Path Length", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="Avg/path")
plt.tight_layout()
plt.savefig("debug_heatmap.png", dpi=300, bbox_inches="tight")
print("\n✓ debug_heatmap.png")
