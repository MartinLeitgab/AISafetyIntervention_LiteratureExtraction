"""
phase2_step4_F4b_pareto_frozenset.py [rev8 — Task #7b]

Adds Pareto-frontier validation to the L2 mechanism-family (frozenset) grouping.
Companion to F3 (body recluster Pareto). Both checks are required for
reviewer-defensible mechanism family extraction.

Pipeline:
  1. Read cooccurrence-families CSV (output of F1 / E3-equivalent on the
     post-recluster path set). Each row is one frozenset of body-cluster IDs.
  2. Build binary vocabulary vector per frozenset.
  3. K-scan via fcluster on Jaccard linkage. Per k, compute:
       - mean within-group Jaccard sim (intra; high = good)
       - max between-group centroid Jaccard sim (inter; low = good)
  4. Pareto choice: smallest k where intra >= INTRA_THRESHOLD AND
     inter <= INTER_THRESHOLD. Fallback: max (intra - inter) gap.
  5. Output Pareto plot + chosen-k CSV. Persist memberships at chosen k.

Inputs:
  step4_cluster_tables/optionB_cooccurrence_families_<suffix>.csv
  step4_cluster_tables/bodysubtype_cluster_representatives_v2.csv (optional, for decoding)

Outputs:
  step4_cluster_tables/frozenset_kscan_metrics_<suffix>.csv
  step4_cluster_tables/frozenset_kscan_chosen_k_<suffix>.csv
  step4_cluster_tables/frozenset_kscan_pareto_<suffix>.png
  step4_cluster_tables/frozenset_groups_pareto_<suffix>.csv
  step4_cluster_tables/frozenset_group_memberships_pareto_<suffix>.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "phase2_results"
STEP4_DIR = RESULTS_DIR / "step4_finalanalysis"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"


def parse_signature(sig_str):
    if not sig_str or pd.isna(sig_str):
        return frozenset()
    return frozenset(p.strip() for p in str(sig_str).split("&"))


def jaccard_centroid(member_X):
    """Compute binary centroid (>=0.5 majority vote) for a group of binary rows."""
    centroid = (member_X.mean(axis=0) >= 0.5).astype(np.float32)
    return centroid


def jaccard_sim(a, b):
    """Jaccard sim between two binary vectors."""
    inter = float(np.dot(a, b))
    union = float(np.sum(np.maximum(a, b)))
    return inter / union if union > 0 else 0.0


def intra_inter_for_k(X, dist_mat, labels):
    """Per-k Pareto metrics:
    intra_mean = mean within-group Jaccard sim across all groups
    inter_max  = max between-centroid Jaccard sim across cluster pairs
    """
    intra_sims = []
    for g in np.unique(labels):
        members = np.where(labels == g)[0]
        if len(members) < 2:
            continue
        sub = dist_mat[np.ix_(members, members)]
        tri = sub[np.triu_indices(len(members), k=1)]
        intra_sims.append(1.0 - tri.mean())
    intra_mean = float(np.mean(intra_sims)) if intra_sims else None
    intra_min = float(np.min(intra_sims)) if intra_sims else None

    centroids = []
    for g in np.unique(labels):
        members = np.where(labels == g)[0]
        centroids.append(jaccard_centroid(X[members]))
    inter_max = None
    if len(centroids) > 1:
        n = len(centroids)
        inter_max = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s = jaccard_sim(centroids[i], centroids[j])
                if s > inter_max:
                    inter_max = s
    return intra_mean, intra_min, inter_max


def choose_k_pareto(metrics_df, intra_threshold, inter_threshold):
    df = metrics_df.sort_values("k")
    pass_mask = (df["intra_mean"] >= intra_threshold) & (
        df["inter_max"] <= inter_threshold
    )
    if pass_mask.any():
        row = df[pass_mask].iloc[0]
        return int(row["k"]), "pass", float(row["intra_mean"]), float(row["inter_max"])
    df = df.copy()
    df["gap"] = df["intra_mean"] - df["inter_max"]
    row = df.sort_values("gap", ascending=False).iloc[0]
    return int(row["k"]), "fail", float(row["intra_mean"]), float(row["inter_max"])


def plot_pareto(
    metrics_df, chosen_k, intra_threshold, inter_threshold, out_path, title
):
    df = metrics_df.sort_values("k")
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")
    sc = ax.scatter(
        df["inter_max"],
        df["intra_mean"],
        c=df["k"],
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"k={int(row['k'])}",
            (row["inter_max"], row["intra_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(
        intra_threshold,
        color="green",
        linestyle="--",
        alpha=0.6,
        label=f"intra >= {intra_threshold}",
    )
    ax.axvline(
        inter_threshold,
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"inter <= {inter_threshold}",
    )
    ax.fill_between(
        [0, inter_threshold],
        intra_threshold,
        1.0,
        alpha=0.10,
        color="green",
        label="Pareto-acceptable",
    )
    cr = df[df["k"] == chosen_k]
    if not cr.empty:
        ax.scatter(
            [cr["inter_max"].iloc[0]],
            [cr["intra_mean"].iloc[0]],
            s=300,
            facecolors="none",
            edgecolors="red",
            linewidths=2.5,
            label=f"chosen k={chosen_k}",
        )
    ax.set_xlabel("max inter-group centroid Jaccard sim (lower = better)")
    ax.set_ylabel("mean intra-group Jaccard sim (higher = better)")
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(top=1.0)
    ax.legend(loc="lower right", fontsize=8)
    plt.colorbar(sc, ax=ax, label="k")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cooccurrence-csv",
        default=str(OUT_TABLES / "optionB_cooccurrence_families_custom_consim1.csv"),
    )
    ap.add_argument(
        "--suffix", default="custom_consim1", help="suffix appended to output files"
    )
    ap.add_argument("--intra-threshold", type=float, default=0.50)
    ap.add_argument("--inter-threshold", type=float, default=0.20)
    ap.add_argument("--k-values", default="3,5,8,10,12,15,18,20,25,30")
    # rev8 §19.3e — filter frozensets by paper-source diversity to remove
    # within-paper DFS combinatorial expansions
    ap.add_argument(
        "--min-paper-sources",
        type=int,
        default=3,
        help="drop frozensets whose n_total_paper_sources < this; 0 disables filter",
    )
    ap.add_argument(
        "--min-RI-pairs",
        type=int,
        default=1,
        help="drop frozensets whose n_distinct_RI_pairs < this; 0 disables filter",
    )
    ap.add_argument(
        "--max-signature-size",
        type=int,
        default=0,
        help="drop frozensets whose signature_size > this; 0 disables",
    )
    args = ap.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]

    print("=" * 70)
    print(f"Phase 2 Step 4 rev8 [Task #7b] — frozenset Pareto check ({args.suffix})")
    print("=" * 70)
    print(f"  cooccurrence_csv  = {args.cooccurrence_csv}")
    print(f"  intra_threshold   = {args.intra_threshold}")
    print(f"  inter_threshold   = {args.inter_threshold}")
    print(f"  k_values          = {k_values}")
    print(f"  min_paper_sources = {args.min_paper_sources}")
    print(f"  min_RI_pairs      = {args.min_RI_pairs}")
    print(f"  max_signature_size= {args.max_signature_size}  (0=disabled)")
    print()

    fam_df_raw = pd.read_csv(args.cooccurrence_csv)
    print(f"Loaded {len(fam_df_raw)} frozensets (pre-filter)")

    # ─── Apply rev8 §19.3e filters ────────────────────────────────────────────
    fam_df = fam_df_raw.copy()
    if args.min_paper_sources > 0 and "n_total_paper_sources" in fam_df.columns:
        before = len(fam_df)
        fam_df = fam_df[fam_df["n_total_paper_sources"] >= args.min_paper_sources]
        print(
            f"  Filter n_total_paper_sources >= {args.min_paper_sources}: "
            f"{before:,} -> {len(fam_df):,}"
        )
    if args.min_RI_pairs > 0 and "n_distinct_RI_pairs" in fam_df.columns:
        before = len(fam_df)
        fam_df = fam_df[fam_df["n_distinct_RI_pairs"] >= args.min_RI_pairs]
        print(
            f"  Filter n_distinct_RI_pairs >= {args.min_RI_pairs}: "
            f"{before:,} -> {len(fam_df):,}"
        )
    if args.max_signature_size > 0 and "signature_size" in fam_df.columns:
        before = len(fam_df)
        fam_df = fam_df[fam_df["signature_size"] <= args.max_signature_size]
        print(
            f"  Filter signature_size <= {args.max_signature_size}: "
            f"{before:,} -> {len(fam_df):,}"
        )
    fam_df = fam_df.reset_index(drop=True)
    print(f"Post-filter: {len(fam_df)} frozensets enter Pareto")
    if len(fam_df) < 3:
        raise SystemExit(
            "ERROR: fewer than 3 frozensets after filtering — "
            "loosen --min-paper-sources or --min-RI-pairs"
        )

    body_reps_path = OUT_TABLES / "bodysubtype_cluster_representatives_v2.csv"
    body_map = {}
    if body_reps_path.exists():
        body_reps = pd.read_csv(body_reps_path)
        body_map = dict(
            zip(body_reps["prefix_key"].astype(str), body_reps["rep_name"].astype(str))
        )
        print(f"Loaded {len(body_map)} body cluster name decodings")

    signatures = [parse_signature(s) for s in fam_df["signature_str"]]
    vocab = sorted(set(k for sig in signatures for k in sig))
    vocab_idx = {k: i for i, k in enumerate(vocab)}
    print(f"  Vocab size: {len(vocab)} body cluster IDs")

    X = np.zeros((len(fam_df), len(vocab)), dtype=np.float32)
    for row_i, sig in enumerate(signatures):
        for k in sig:
            if k in vocab_idx:
                X[row_i, vocab_idx[k]] = 1.0

    dist_vec = pdist(X, metric="jaccard")
    dist_mat = squareform(dist_vec)
    Z = linkage(dist_vec, method="average")
    print(
        f"  Pairwise Jaccard distance: mean={dist_vec.mean():.3f}, "
        f"median={np.median(dist_vec):.3f}"
    )

    rows = []
    labels_per_k = {}
    print("\n--- K-scan (intra/inter Jaccard) ---")
    for k in k_values:
        if k >= len(fam_df):
            print(f"  k={k}: skipped (>= n_frozensets={len(fam_df)})")
            continue
        labels = fcluster(Z, t=k, criterion="maxclust")
        labels_per_k[k] = labels
        intra_mean, intra_min, inter_max = intra_inter_for_k(X, dist_mat, labels)
        n_groups = len(np.unique(labels))
        rows.append(
            {
                "k": k,
                "n_frozensets": len(fam_df),
                "n_groups_realized": n_groups,
                "intra_mean": round(intra_mean, 4) if intra_mean is not None else None,
                "intra_min": round(intra_min, 4) if intra_min is not None else None,
                "inter_max": round(inter_max, 4) if inter_max is not None else None,
            }
        )
        if intra_mean is None or intra_min is None or inter_max is None:
            print(
                f"  k={k:3d}: n_groups={n_groups:3d} | DEGENERATE — no group has >=2 members "
                f"(intra_mean={intra_mean}, intra_min={intra_min}, inter_max={inter_max})"
            )
        else:
            print(
                f"  k={k:3d}: n_groups={n_groups:3d} | intra_mean={intra_mean:.4f} "
                f"intra_min={intra_min:.4f} inter_max={inter_max:.4f}"
            )

    metrics_df = pd.DataFrame(rows)
    metrics_csv = OUT_TABLES / f"frozenset_kscan_metrics_{args.suffix}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nWritten: {metrics_csv.name}")

    chosen_k, status, intra_chosen, inter_chosen = choose_k_pareto(
        metrics_df, args.intra_threshold, args.inter_threshold
    )
    chosen_df = pd.DataFrame(
        [
            {
                "suffix": args.suffix,
                "chosen_k": chosen_k,
                "status": status,
                "intra_mean": round(intra_chosen, 4),
                "inter_max": round(inter_chosen, 4),
                "gap": round(intra_chosen - inter_chosen, 4),
                "intra_threshold": args.intra_threshold,
                "inter_threshold": args.inter_threshold,
            }
        ]
    )
    chosen_csv = OUT_TABLES / f"frozenset_kscan_chosen_k_{args.suffix}.csv"
    chosen_df.to_csv(chosen_csv, index=False)
    verdict = (
        "OK PARETO PASS"
        if status == "pass"
        else "** PARETO FAIL (frozenset diversity at L2 too high; report as finding) **"
    )
    print(
        f"\nChosen k={chosen_k}  status={status}  "
        f"intra={intra_chosen:.4f} inter={inter_chosen:.4f}  {verdict}"
    )

    plot_path = OUT_TABLES / f"frozenset_kscan_pareto_{args.suffix}.png"
    plot_pareto(
        metrics_df,
        chosen_k,
        args.intra_threshold,
        args.inter_threshold,
        plot_path,
        f"L2 Frozenset Pareto frontier — {args.suffix}",
    )
    print(f"Wrote {plot_path.name}")

    # Persist groups at chosen k
    chosen_labels = labels_per_k[chosen_k]
    group_rows = []
    membership_rows = []

    def decode_sig(sig):
        return " | ".join(body_map.get(k, k) for k in sorted(sig))

    for g in sorted(np.unique(chosen_labels)):
        members = np.where(chosen_labels == g)[0]
        n_paths_total = (
            int(fam_df["n_paths"].values[members].sum()) if "n_paths" in fam_df else 0
        )
        sub_X = X[members]
        top_idx = np.argsort(sub_X.mean(axis=0))[::-1][:8]
        centroid_components = [vocab[i] for i in top_idx if sub_X.mean(axis=0)[i] > 0]
        centroid_decoded = " | ".join(
            body_map.get(k, k) for k in centroid_components[:5]
        )
        sub_dist = dist_mat[np.ix_(members, members)]
        if len(members) > 1:
            tri = sub_dist[np.triu_indices(len(members), k=1)]
            intra_g = float(1 - tri.mean())
        else:
            intra_g = 1.0
        group_rows.append(
            {
                "group_id": int(g),
                "n_frozensets": len(members),
                "n_paths_total": n_paths_total,
                "centroid_components": " & ".join(centroid_components[:5]),
                "centroid_decoded": centroid_decoded,
                "intra_jaccard_mean": round(intra_g, 4),
            }
        )
        for local_i, global_i in enumerate(members):
            membership_rows.append(
                {
                    "family_id": int(fam_df["family_id"].iloc[global_i])
                    if "family_id" in fam_df
                    else int(global_i),
                    "group_id": int(g),
                    "n_paths": int(fam_df["n_paths"].iloc[global_i])
                    if "n_paths" in fam_df
                    else 0,
                    "signature_str": str(fam_df["signature_str"].iloc[global_i]),
                }
            )

    groups_csv = OUT_TABLES / f"frozenset_groups_pareto_{args.suffix}.csv"
    pd.DataFrame(group_rows).sort_values("n_paths_total", ascending=False).to_csv(
        groups_csv, index=False
    )
    members_csv = OUT_TABLES / f"frozenset_group_memberships_pareto_{args.suffix}.csv"
    pd.DataFrame(membership_rows).to_csv(members_csv, index=False)
    print(f"Wrote {groups_csv.name} ({len(group_rows)} groups)")
    print(f"Wrote {members_csv.name} ({len(membership_rows)} memberships)")
    print("\nDone.")


if __name__ == "__main__":
    main()
