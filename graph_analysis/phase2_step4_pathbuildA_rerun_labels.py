"""
PathbuildA label rerun — corrected qualifying filter.

Uses the existing (correct) optionA_kmeans_model.pkl (fitted on full data, April 5)
to predict cluster labels for qualifying consim1 paths ONLY:
  - intervention endpoint maturity >= 3
  - max consecutive SIM hops <= 1  (consim1)

This brings PathbuildA onto the same 75,008-path qualifying set used by PathbuildB,
making both chain-clustering approaches directly comparable.

Previous runs processed all ~1,054,440 unconstrained paths (no quality filters) —
that was incorrect.

Outputs:
  step4_finalanalysis/optionA_cluster_labels.pkl   (corrected, consim1 qualifying)
  step4_metaclusters/pathbuildA_intracentroid_histograms.png  (B2 rerun)
  step4_metaclusters/pathbuildA_intracentroid_stats.csv       (B2 rerun)
"""

import io
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
PROJECT_ROOT = BASE.parent
STEP1_DIR = (
    PROJECT_ROOT
    / "graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
)
PATHS_FILE = (
    PROJECT_ROOT
    / "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
)
ARCHIVE = BASE / "phase2_results/step4_finalanalysis/archive_rev3"
STEP4_DIR = BASE / "phase2_results/step4_finalanalysis"
META_OUT = BASE / "phase2_results/step4_finalanalysis/step4_metaclusters"
META_OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5000


# ── Safe loader for cross-version numpy pickle compatibility ──────────────
class _DummyRNG:
    """Absorbs numpy random-state __setstate__ across version boundaries."""

    def __setstate__(self, state):
        pass

    def __reduce__(self):
        return (_DummyRNG, ())


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.random._pickle" and name in (
            "__bit_generator_ctor",
            "__randomstate_ctor",
            "__generator_ctor",
        ):
            return lambda *a, **kw: _DummyRNG()
        if "mt19937" in module.lower() or "randomstate" in name.lower():
            return lambda *a, **kw: _DummyRNG()
        return super().find_class(module, name)


def safe_pickle_load(path):
    with open(path, "rb") as f:
        data = f.read()
    return _SafeUnpickler(io.BytesIO(data)).load()


# ── 1. Load correct KMeans model (April 5 full-run) ───────────────────────
print("Loading optionA_kmeans_model.pkl (April 5 full-run) …", flush=True)
kmeans = safe_pickle_load(ARCHIVE / "optionA_kmeans_model.pkl")
print(
    f"  {kmeans.n_clusters} clusters, centroids shape: {kmeans.cluster_centers_.shape}",
    flush=True,
)

centroids = kmeans.cluster_centers_
centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

# ── 2. Load node embeddings ────────────────────────────────────────────────
print("Loading graph_node_attributes.pkl …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs: dict = pickle.load(f)
print(f"  {len(node_attrs):,} nodes  ({time.time() - t0:.1f}s)", flush=True)


def parse_embedding(emb) -> np.ndarray | None:
    if emb is None:
        return None
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32)
    if isinstance(emb, str):
        return np.fromstring(emb.strip("<>"), sep=", ").astype(np.float32)
    return np.array(emb, dtype=np.float32)


# Pre-build embedding cache
print("Pre-building embedding cache …", flush=True)
t0 = time.time()
emb_cache: dict[int, np.ndarray] = {}
for nid, attrs in node_attrs.items():
    emb_raw = attrs.get("embedding")
    if emb_raw is not None:
        try:
            emb_cache[int(nid)] = parse_embedding(emb_raw)
        except Exception:
            pass
print(f"  {len(emb_cache):,} nodes cached  ({time.time() - t0:.1f}s)", flush=True)


# ── 3. Build sim_edge_set + VPN for consim1 filter ────────────────────────
# Must match PathbuildB exactly: build VPN from maturity>=3 paths first,
# then restrict SIM edges to VPN-pair edges only.
def cos_sim_from_score(s: float) -> float:
    return 1.0 - float(s) ** 2 / 2.0


print("Loading graph_edge_data.pkl for consim1 filter …", flush=True)
t0 = time.time()
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data: list = pickle.load(f)
print(f"  {len(edge_data):,} edges  ({time.time() - t0:.1f}s)", flush=True)

# Build unconstrained VPN: all nodes on any path where path[-1] has maturity>=3
print("Building unconstrained VPN (maturity>=3) …", flush=True)
t0 = time.time()
vpn_broad: set[int] = set()
with open(PATHS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        path = [int(x) for x in (obj.get("path") or obj.get("node_id_sequence") or [])]
        if len(path) < 1:
            continue
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            vpn_broad.update(path)
print(f"  {len(vpn_broad):,} VPN nodes  ({time.time() - t0:.1f}s)", flush=True)

# Build sim_edge_set restricted to VPN pairs (SIM >= 0.9 only)
print("Building sim_edge_set (SIM>=0.9, VPN-restricted) …", flush=True)
t0 = time.time()
sim_edge_set: set[tuple[int, int]] = set()
for e in edge_data:
    if str(e.get("type", "")).upper() == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s2, t2 = int(e["source"]), int(e["target"])
                if s2 in vpn_broad and t2 in vpn_broad:
                    sim_edge_set.add((min(s2, t2), max(s2, t2)))
            except (ValueError, TypeError):
                pass
print(f"  {len(sim_edge_set):,} sim edges  ({time.time() - t0:.1f}s)", flush=True)
del edge_data, vpn_broad  # free memory


def max_consec_sim(path_ids: list[int]) -> int:
    """Count the maximum run of consecutive SIM edges in a path."""
    max_run = run = 0
    for i in range(len(path_ids) - 1):
        a, b = path_ids[i], path_ids[i + 1]
        if (min(a, b), max(a, b)) in sim_edge_set:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


# ── 4. Scan qualifying paths (maturity>=3 AND consim1) ────────────────────
print(f"Scanning {PATHS_FILE.name} (consim1 qualifying paths only) …", flush=True)
t0 = time.time()
all_records = []  # list of (mean_body_emb, body_ids, full_path)
n_total = n_maturity_pass = n_fit = 0

with open(PATHS_FILE) as f:
    for lno, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        path = [int(x) for x in (obj.get("path") or obj.get("node_id_sequence") or [])]
        n_total += 1
        if len(path) < 3:
            continue
        interv_id = path[-1]
        # Filter 1: intervention maturity >= 3
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) < 3:
            continue
        n_maturity_pass += 1
        # Filter 2: consim1 — max consecutive SIM hops <= 1
        if max_consec_sim(path) > 1:
            continue
        body_ids = path[1:-1]
        embs = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
        if not embs:
            continue
        mean_emb = np.stack(embs).mean(axis=0).astype(np.float32)
        all_records.append((mean_emb, body_ids, path))
        n_fit += 1
        if n_fit % 10_000 == 0:
            print(
                f"  … {n_fit:,} qualifying records  ({time.time() - t0:.0f}s)",
                flush=True,
            )

print(
    f"  Scan done: {n_total:,} total | {n_maturity_pass:,} maturity>=3 | "
    f"{n_fit:,} consim1 qualifying  ({time.time() - t0:.1f}s)",
    flush=True,
)

# ── 5. Predict labels ─────────────────────────────────────────────────────
print("Predicting labels …", flush=True)
t0 = time.time()
all_embs = np.stack([r[0] for r in all_records], dtype=np.float32)
labels = kmeans.predict(all_embs)
print(f"  Predicted {len(labels):,} labels  ({time.time() - t0:.1f}s)", flush=True)

# ── 6. Save corrected PKL ─────────────────────────────────────────────────
out_pkl = STEP4_DIR / "optionA_cluster_labels.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump(
        {"labels": labels, "records": [(r[1], r[2]) for r in all_records]},
        f,
    )
print(f"Saved {out_pkl}  ({len(labels):,} records)", flush=True)

# Label distribution
unique, counts = np.unique(labels, return_counts=True)
print("Label distribution (top 10 clusters by count):")
for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
    print(f"  C{cid:2d}: {cnt:,}")

# ── 6. Rerun B2 intra-centroid histograms ─────────────────────────────────
print("\nRerunning B2 PathbuildA intra-centroid histograms …", flush=True)

cluster_sims: dict[int, list[float]] = {c: [] for c in range(kmeans.n_clusters)}

for i, (record, label) in enumerate(zip(all_records, labels)):
    mean_emb_norm = record[0] / (np.linalg.norm(record[0]) + 1e-12)
    cos_sim = float(np.dot(mean_emb_norm, centroids_norm[label]))
    cluster_sims[int(label)].append(cos_sim)

# Stats CSV
rows = []
for c in range(kmeans.n_clusters):
    sims = cluster_sims[c]
    if sims:
        rows.append(
            {
                "cluster_id": c,
                "n_paths": len(sims),
                "mean_cosine_sim": float(np.mean(sims)),
                "median_cosine_sim": float(np.median(sims)),
                "std_cosine_sim": float(np.std(sims)),
                "min_cosine_sim": float(np.min(sims)),
                "max_cosine_sim": float(np.max(sims)),
            }
        )
    else:
        rows.append(
            {
                "cluster_id": c,
                "n_paths": 0,
                "mean_cosine_sim": float("nan"),
                "median_cosine_sim": float("nan"),
                "std_cosine_sim": float("nan"),
                "min_cosine_sim": float("nan"),
                "max_cosine_sim": float("nan"),
            }
        )

stats_df = pd.DataFrame(rows)
stats_df.to_csv(META_OUT / "pathbuildA_intracentroid_stats.csv", index=False)
overall_mean = stats_df["mean_cosine_sim"].mean()
print(
    f"  Overall mean cosine sim: {overall_mean:.4f}  (was 0.8014 on 10k sample)",
    flush=True,
)

# 40-histogram canvas (8×5)
NCOLS, NROWS = 8, 5
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(24, 14))
axes_flat = axes.flatten()

for c in range(40):
    ax = axes_flat[c]
    sims = cluster_sims[c]
    n = len(sims)
    if n == 0:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(f"C{c}", fontsize=8)
        continue
    mean_s = float(np.mean(sims))
    ax.hist(sims, bins=30, color="coral", edgecolor="none", alpha=0.85)
    ax.axvline(mean_s, color="darkred", linewidth=1.2, linestyle="--")
    ax.set_title(f"C{c}  μ={mean_s:.3f}  n={n:,}", fontsize=7.5)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("cos sim to centroid", fontsize=6)
    ax.set_ylabel("count", fontsize=6)

for i in range(40, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle(
    f"PathbuildA Chain Clusters — Intra-centroid Cosine Similarity Distributions\n"
    f"(40 KMeans clusters on mean chain-body embeddings; n={n_fit:,} paths; overall mean={overall_mean:.3f})\n"
    f"Note: wider/lower distributions expected — body spans multiple concept subtypes",
    fontsize=11,
    y=1.01,
)
plt.tight_layout()
out_hist = META_OUT / "pathbuildA_intracentroid_histograms.png"
fig.savefig(out_hist, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_hist}", flush=True)
print("Done — PathbuildA labels + B2 rerun complete.", flush=True)
