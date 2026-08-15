#!/usr/bin/env python
"""Are the seven schema stages separable in the content the model wrote for them?

Reviewer question (NeurIPS review W20/Q6, 2026-08-14): the five intermediate stages are
the paper's novel object, and no evidence is offered that they are separable rather than
a labelling convention the prompt imposes.

This is the Class B half of that question. It does NOT ask whether a second model or a
human would assign the same stage to the same content -- that needs an annotation run.
It asks the prior question: does the TEXT the extractor wrote under each stage label
carry a signal that identifies the label? If a probe cannot recover the stage from the
node's own text, the stage vocabulary is decoration. If it can, the labels track
something in the content, whether or not that something is the right thing.

Method
------
Held-out probe on the released node embeddings (text-embedding-3-small, 1536d, over node
name + aliases + description -- the node's own text, never its graph position):

  * stratified sample, capped per class so a frequent class cannot carry the score;
  * split by SOURCE DOCUMENT, not by node, so no test node shares a paper with a train
    node (nodes from one paper are written in one call and are heavily correlated);
  * multinomial logistic regression, L2, on L2-normalised embeddings;
  * reported against two baselines: uniform chance and the majority class.

Two tasks: the five intermediate stages (the contested ones), and all seven including
the risk and intervention endpoints (which the schema pins to path position, so they are
expected to be easier and are reported as a reference point).

Also reported, as a descriptive that needs no classifier: mean cosine of each class to
its own centroid against its mean cosine to the other classes' centroids.

Decomposition (added after the first run returned 0.988, which is too high to take at
face value). The extraction prompt prescribes a NAMING TEMPLATE per stage --- risk is
"[Phenomenon/Problem] in [Context]", validation evidence is "[Measurement and Result] in
[Context]", and so on -- so a probe may be recovering the template rather than any
distinction between the stages. Three text-only baselines separate the two readings:
word-level TF-IDF on the node NAME alone, on the DESCRIPTION alone, and on both. If the
name alone already scores near the embedding probe, the signal is the naming convention
the prompt imposes, and the result must be reported as such.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_review_stage_separability.py

Output: graph_analysis/phase2_results/experiment_review_stage_separability_report.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
NODE_PKL = STEP1 / "graph_node_attributes.pkl"
OUT = ROOT / "phase2_results/experiment_review_stage_separability_report.json"

# Canonical logical-chain order. Never alphabetise these.
BODY = [
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]
ALL7 = ["risk"] + BODY + ["intervention"]

PER_CLASS_CAP = 6000  # nodes per class, before the document-level split
TEST_DOC_FRACTION = 0.30
SEED = 42


def fail(msg: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n"
        f"  expected artifact: {NODE_PKL}\n"
        "  produced by: intervention_graph_creation ingestion + "
        "phase2_step1_loadandparse.py against the FalkorDB dump\n"
        "  this script does NOT fall back to a cached or slim node file: the slim file "
        "carries no embeddings, and a probe without embeddings is not this measurement."
    )


def parse_embedding(raw):
    """Embeddings are stored as '<f, f, ...>' strings on the node attributes."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, np.ndarray)):
        return np.asarray(raw, dtype=np.float32)
    s = str(raw).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    try:
        return np.fromstring(s, dtype=np.float32, sep=",")
    except ValueError:
        return None


def class_of(attrs):
    if (attrs.get("type") or "").lower() == "intervention":
        return "intervention"
    return (attrs.get("concept_category") or "").lower().strip()


def probe(X, y, doc, classes, rng):
    """Document-disjoint train/test split, then a multinomial logistic probe."""
    docs = np.array(sorted(set(doc)))
    rng.shuffle(docs)
    n_test = int(round(TEST_DOC_FRACTION * len(docs)))
    test_docs = set(docs[:n_test].tolist())
    is_test = np.array([d in test_docs for d in doc])

    Xtr, ytr = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    if len(set(yte.tolist())) < len(classes):
        fail("a class is absent from the held-out documents; raise PER_CLASS_CAP")

    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    majority = Counter(ytr.tolist()).most_common(1)[0][0]
    cm = confusion_matrix(yte, pred, labels=list(range(len(classes))))
    per_class = f1_score(yte, pred, labels=list(range(len(classes))), average=None)
    return {
        "n_classes": len(classes),
        "classes_in_logical_chain_order": classes,
        "n_train_nodes": int(len(ytr)),
        "n_test_nodes": int(len(yte)),
        "n_train_documents": int(len(docs) - n_test),
        "n_test_documents": int(n_test),
        "accuracy": round(float((pred == yte).mean()), 4),
        "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "per_class_f1": {c: round(float(v), 4) for c, v in zip(classes, per_class)},
        "baseline_uniform_chance_accuracy": round(1.0 / len(classes), 4),
        "baseline_majority_class_accuracy": round(float((yte == majority).mean()), 4),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
    }


def text_probe(texts, y, doc, classes, label):
    """Same document-disjoint split, word TF-IDF instead of the embedding.

    Separates 'the stages are distinguishable' from 'the prompt names them distinctly'.
    """
    docs = np.array(sorted(set(doc)))
    np.random.default_rng(SEED).shuffle(docs)
    test_docs = set(docs[: int(round(TEST_DOC_FRACTION * len(docs)))].tolist())
    is_test = np.array([d in test_docs for d in doc])

    vec = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, min_df=3, max_features=50000
    )
    Xtr = vec.fit_transform([t for t, m in zip(texts, is_test) if not m])
    Xte = vec.transform([t for t, m in zip(texts, is_test) if m])
    ytr, yte = y[~is_test], y[is_test]

    clf = LogisticRegression(max_iter=3000, C=4.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "field": label,
        "n_features": int(Xtr.shape[1]),
        "accuracy": round(float((pred == yte).mean()), 4),
        "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "per_class_f1": {
            c: round(float(v), 4)
            for c, v in zip(
                classes,
                f1_score(yte, pred, labels=list(range(len(classes))), average=None),
            )
        },
    }


def centroid_separation(X, y, classes):
    """Mean cosine to own centroid vs mean cosine to the other centroids."""
    cents = np.stack(
        [
            X[y == i].mean(axis=0) / np.linalg.norm(X[y == i].mean(axis=0))
            for i in range(len(classes))
        ]
    )
    sims = X @ cents.T
    out = {}
    for i, c in enumerate(classes):
        m = y == i
        own = float(sims[m, i].mean())
        others = float(np.delete(sims[m], i, axis=1).mean())
        out[c] = {
            "mean_cosine_to_own_centroid": round(own, 4),
            "mean_cosine_to_other_centroids": round(others, 4),
            "margin": round(own - others, 4),
        }
    return out


def main():
    t0 = time.time()
    if not NODE_PKL.exists():
        fail("node attributes checkpoint not found")
    print(
        "loading node attributes (about 3.3 GB, expect a few minutes) ...", flush=True
    )
    na = pickle.load(open(NODE_PKL, "rb"))
    print(f"  {len(na)} nodes", flush=True)

    rng = np.random.default_rng(SEED)
    by_class = defaultdict(list)
    for nid, a in na.items():
        c = class_of(a)
        if c in ALL7 and a.get("url"):
            by_class[c].append(nid)
    for c in ALL7:
        if not by_class[c]:
            fail(f"no nodes found for class {c!r}")

    sampled = []
    for c in ALL7:
        ids = np.array(by_class[c])
        rng.shuffle(ids)
        sampled.extend((int(i), c) for i in ids[:PER_CLASS_CAP])
    print(f"sampled {len(sampled)} nodes; parsing embeddings ...", flush=True)

    vecs, labels, docs, names, descs = [], [], [], [], []
    n_bad = 0
    for nid, c in sampled:
        v = parse_embedding(na[nid].get("embedding"))
        if v is None or v.shape[0] != 1536 or not np.isfinite(v).all():
            n_bad += 1
            continue
        vecs.append(v)
        labels.append(c)
        docs.append(na[nid]["url"])
        names.append(str(na[nid].get("name") or ""))
        descs.append(str(na[nid].get("description") or ""))
    if n_bad:
        print(f"  skipped {n_bad} nodes with an unusable embedding", flush=True)
    if n_bad > 0.01 * len(sampled):
        fail(
            f"{n_bad} unusable embeddings in the sample -- the checkpoint is not intact"
        )

    X = np.stack(vecs)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    docs = np.array(docs)
    print(f"  matrix {X.shape}, {len(set(docs.tolist()))} source documents", flush=True)

    report = {
        "experiment": "stage separability of the extraction schema (reviewer W20/Q6)",
        "question": (
            "Does the text the extractor wrote under each stage label carry a signal "
            "that identifies the label, under a document-disjoint split?"
        ),
        "SCOPE_NOTE": (
            "This measures separability of the extractor's OWN output, not agreement "
            "with a second annotator. A high score means the stage labels track "
            "something stable in the text the model wrote; it is not evidence that the "
            "stage assignment is correct, and it cannot be, since the same model wrote "
            "both the text and the label. Cross-annotator agreement remains unmeasured."
        ),
        "inputs": {"node_attributes": str(NODE_PKL)},
        "sampling": {
            "per_class_cap": PER_CLASS_CAP,
            "seed": SEED,
            "test_document_fraction": TEST_DOC_FRACTION,
            "split_unit": "source document (URL), so no test node shares a paper with a train node",
            "features": "released node embedding (text-embedding-3-small, 1536d) over "
            "name + aliases + description; no graph structure of any kind",
            "n_nodes_sampled": int(X.shape[0]),
            "n_documents": int(len(set(docs.tolist()))),
            "class_counts": dict(Counter(labels)),
        },
    }

    for name, classes in [
        ("five_intermediate_stages", BODY),
        ("all_seven_stages", ALL7),
    ]:
        keep = np.array([lbl in classes for lbl in labels])
        idx = {c: i for i, c in enumerate(classes)}
        y = np.array([idx[lbl] for lbl in np.array(labels)[keep]])
        print(f"probing {name} on {keep.sum()} nodes ...", flush=True)
        report[name] = probe(
            X[keep], y, docs[keep], classes, np.random.default_rng(SEED)
        )
        report[name]["centroid_separation"] = centroid_separation(X[keep], y, classes)
        nm = [t for t, m in zip(names, keep) if m]
        ds = [t for t, m in zip(descs, keep) if m]
        report[name]["lexical_ablation"] = {
            "WHY": "The prompt prescribes a naming template per stage. If the node NAME "
            "alone recovers the label, the probe is reading that template and not a "
            "distinction between the stages.",
            "name_only": text_probe(nm, y, docs[keep], classes, "name"),
            "description_only": text_probe(ds, y, docs[keep], classes, "description"),
            "name_and_description": text_probe(
                [a + " " + b for a, b in zip(nm, ds)],
                y,
                docs[keep],
                classes,
                "name+description",
            ),
        }

    report["wall_clock_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    for name in ("five_intermediate_stages", "all_seven_stages"):
        r = report[name]
        print(
            f"\n{name}: accuracy {r['accuracy']:.3f} "
            f"(chance {r['baseline_uniform_chance_accuracy']:.3f}, "
            f"majority {r['baseline_majority_class_accuracy']:.3f}), "
            f"macro-F1 {r['macro_f1']:.3f}"
        )
        for c, v in r["per_class_f1"].items():
            print(f"    {c:<26} F1 {v:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
