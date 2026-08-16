#!/usr/bin/env python
"""Does the chain survive when the schema is removed, or when the argument is?

GitHub issue #165. OPEN_ITEMS.md S6+S8 (merged), runbook R2. Reviewer comments R29 / R30,
raised by four of six external reviewers at both bars.

The manuscript names two fidelity controls in Limitations and runs neither:

  schema ablation      -- 87.4% of chains carry all five intermediate stages, which is what
                          schema-filling predicts as much as what faithful extraction
                          predicts. Nothing separates the two.
  degraded source      -- app:failure shows ONE chain invented from an expository article
                          about prime numbers. An existence proof, not a rate.

Four arms over ONE document sample, so every arm is paired with its own baseline:

  A  released extraction, as shipped          no call; read from the released graph
  E  prompt with the five stages removed      does the structure survive un-prompted?
  F  released prompt, sentence-shuffled       confabulation from topical vocabulary?
  G  released prompt, reference list only     the same, harder

Scoring is STRUCTURAL and uses no judge. Each arm's response is parsed, the per-document
graph is rebuilt, and the released enumerator's constraints are applied unchanged:

    edge confidence >= 3 on every edge traversed, endpoint intervention maturity >= 3,
    first hop on an intermediate subtype, simple paths, stop at the first qualifying
    intervention, 3 <= hops <= 30

which is exactly phase2_step4_F2v4_hopwise_falkordb.py's rule set, re-implemented here
against a single paper's graph rather than the whole database.

Arm E emits free-form category labels by construction. They are mapped onto the five
stages by ONE rubric-driven call over the pooled distinct labels; the rubric is written
into the receipt verbatim so a reader can disagree with it.

DELIBERATE, and stated in the receipt rather than left to be discovered:
  * The extractor is o3 at reasoning effort medium -- the corpus extractor -- through the
    same /v1/responses request shape extractor.py builds. A Claude-tier proxy would make
    arms E and F statements about a model that never produced the corpus.
  * Transport is SYNCHRONOUS where the corpus run used the batch API. Model, prompt and
    reasoning effort are identical; the difference is billing and latency. Batch wall-clock
    is unbounded and this is a 90-call job.
  * Every response is written to disk as it arrives. A mid-run kill loses one call.

CLASS A (spends metered OpenAI tokens). Run from graph_analysis/:

    python -u experiment_review_schema_ablation.py                 # dry run, no API call
    python -u experiment_review_schema_ablation.py --run E,F,G     # execute
    python -u experiment_review_schema_ablation.py --score         # receipt from responses

Output:
    phase2_results/ablation_raw/<arm>/<doc_id>.json                one file per call
    phase2_results/experiment_review_schema_ablation_report.json   the receipt
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken

ROOT = Path(__file__).parent
REPO = ROOT.parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
PATHS = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
ARD = REPO / "intervention_graph_creation/data/raw/ard_json_full"
PROMPT_PY = REPO / "intervention_graph_creation/src/prompt/final_primary_prompt.py"
RAW_DIR = ROOT / "phase2_results/ablation_raw"
OUT = ROOT / "phase2_results/experiment_review_schema_ablation_report.json"

# The key is read from another project's .env at run time and never copied into this repo,
# never logged, and never written into a receipt. This repo is public.
ENV_PATH = Path.home() / "0_project_work/ExistentialRiskBenchmark/.env"

MODEL = "o3"
REASONING_EFFORT = "medium"
N_DOCS = 30
N_DOCS_G = 25  # arm G's own population; see build_payloads
SEED = 42
MAX_WORKERS = 4
TIMEOUT_S = 900
ENCODING = "o200k_base"

# Rates are INPUTS to this script, printed with the result so a stale rate is visible.
# SYNCHRONOUS o3 rates, USD per million tokens; this run is not a batch run.
RATE_IN, RATE_OUT = 2.00, 8.00
REASONING_RATIOS = [0.0, 1.0, 2.0, 4.0]

# The five stage names, and the step_1 phrasings that name them without using the labels.
# The arm-E prompt must contain none of them.
STAGE_WORDS = [
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]
STAGE_ABBR = {
    "problem analysis": "pa",
    "theoretical insight": "ti",
    "design rationale": "dr",
    "implementation mechanism": "im",
    "validation evidence": "va",
}
CHAIN_ORDER = ["risk"] + STAGE_WORDS + ["intervention"]

EDGE_CONFIDENCE_MIN = 3
INTERVENTION_MATURITY_MIN = 3
MIN_HOPS, MAX_HOPS = 3, 30


def fail(msg: str, artifact: Path | str, produced_by: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n"
        f"  expected artifact: {artifact}\n"
        f"  produced by: {produced_by}\n"
        "  this script does NOT substitute a smaller sample, a cached response or a "
        "different model for a missing input."
    )


# --------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------


def load_released_prompt() -> str:
    if not PROMPT_PY.exists():
        fail("extraction prompt module not found", PROMPT_PY, "the extraction pipeline")
    src = PROMPT_PY.read_text(encoding="utf-8")
    m = re.search(r'PROMPT_EXTRACT\s*=\s*"""(.*?)"""', src, re.S)
    if not m:
        fail("PROMPT_EXTRACT string not found", PROMPT_PY, "the extraction pipeline")
    return m.group(1)


def build_schema_blind_prompt(prompt: str) -> str:
    """Remove the five stages from the released prompt, changing nothing else.

    Surgery on the released text rather than a prompt written from scratch: arm E has to
    differ from arm A in ONE respect, and a re-written prompt would differ in many. Each
    replacement is asserted, so a future edit to final_primary_prompt.py that moves this
    text makes the script fail rather than silently ablate nothing.
    """
    out = prompt
    subs = [
        # step_1: the five stages are named in prose here even though the labels are not used.
        (
            """Read data source completely and identify:
1. Core risks addressed
2. **Underlying assumptions and theoretical insights**
3. **Design principles and reasoning steps justifying solutions**
4. **Specific mechanisms enabling frameworks to work**
5. Main findings and proposed interventions
6. Identify all major causal-interventional pathways through the data source silently, without outputting anything yet""",
            """Read data source completely and identify:
1. Core risks addressed
2. Proposed interventions
3. Whatever reasoning the data source itself puts between a risk and an intervention, in
   the source's own terms and at whatever level of detail the source supplies
4. Identify all major causal-interventional pathways through the data source silently, without outputting anything yet""",
        ),
        # step_2: the six-category vocabulary and its name templates.
        (
            """### Concept Node Categories & Name Patterns in preferred order of causal-interventional flow**

1. **Risk**: "[Canonical Specific Phenomenon/Problem Name] in [Context]"
2. **Problem Analysis**: "[Mechanism Causing Risk] in [Context]"
3. **Theoretical Insight**: "[Assumption/Hypothesized Resolution Opportunity of Problem/Claim] in [Context]"
4. **Design Rationale**: "[Solution Approach to Resolve Problem] in [Context]"
5. **Imlpementation Mechanism**: "[Technique/Implementation of Approach] in [Context]"
6. **Validation Evidence**: "[Measurement and Result of Approach] in [Context]"

Capture details of each node in the node description attribute, e.g. a summary of detailed findings for a validation evidence node""",
            """### Concept Node Categories

Concept nodes carry a category label. Two labels are fixed because they are the endpoints
of every path: "risk" for the phenomenon or problem a path starts from, and the intervention
node type for what a path ends at. For every concept node BETWEEN those endpoints, choose
your own short category label describing the role that node plays in the source's argument,
and use your labels consistently across the whole data source. Name concept nodes as
"[Specific Phenomenon or Claim] in [Context]".

Capture details of each node in the node description attribute.""",
        ),
        # step_4: the mandated flow through the six categories.
        (
            """**Putting it all together**: Every knowledge fabric path should start with a risk node, flow through the six concept node categories defined above, and end with an intervention node as closely as possible.
- DO NOT connect risk nodes directly to intervention nodes- ALWAYS build the reasoning path between risk and interventions nodes with the six concept node categories.""",
            """**Putting it all together**: Every knowledge fabric path should start with a risk node, flow through however many intermediate concept nodes the source's own argument supports, and end with an intervention node.
- DO NOT connect risk nodes directly to intervention nodes- ALWAYS build the reasoning path between risk and intervention nodes out of intermediate concept nodes.""",
        ),
        (
            """- If the required flow and succession of node types/categories is not explicitly supported by the data source, use moderate inference to construct knowledge fabric paths as close to this intent as possible and mark appropriately in edge confidence and edge rationale where inference was used (confidence must be 1 or 2 with inference).
- Multiple nodes with the same category can exist in the reasoning path if concept richness as presented in the data source warrants more refinement.""",
            """- If a path from a risk to an intervention is not explicitly supported by the data source, use moderate inference to construct it and mark appropriately in edge confidence and edge rationale where inference was used (confidence must be 1 or 2 with inference).
- Multiple nodes with the same category can exist in the reasoning path if concept richness as presented in the data source warrants more refinement.""",
        ),
        # step_4: the worked seven-node template, which is the schema by example.
        (
            """<Knowledge_Fabric_Path_Template>
**Always start at risk node, always flow through all intermediate nodes, and always end at intervention node**
Start node (concept: risk) "Gradual disempowerment of humans by AI systems" → edge "caused_by" →
node (concept:problem analysis) "Sycophantic behavior in LLMs" → edge "caused_by" →
node (concept:theoretical insight) "Systematic biases in human feedback" → edge "mitigated_by" →
node (concept:design rationale) "AI self-evaluation reducing human feedback dependency" → edge "implemented_by" →
node (concept:implementation mechanism) "Constitutional principles as bias-free training signal" → edge "validated_by"
node (concept:validation evidence) "Sycophancy evaluation benchmark improvement" → edge "motivates" →
end node (intervention) "Fine-tune/RL train models with constitutional AI to reduce sycophantic responses"
</Knowledge_Fabric_Path_Template>""",
            """<Knowledge_Fabric_Path_Template>
**Always start at a risk node, always flow through intermediate concept nodes, and always end at an intervention node**
Start node (concept: risk) "[a phenomenon or problem] in [context]" → edge →
one or more intermediate concept nodes, each carrying a category label you have chosen for
the role it plays in the argument, in whatever order and number the data source supports → edge →
end node (intervention) "[the action the data source proposes]"
</Knowledge_Fabric_Path_Template>""",
        ),
        # step_3: the edge-type list names a stage in its parenthetical.
        (
            "motivates (use 'motivates' from validation evidence to intervention)",
            "motivates (use 'motivates' for the edge that enters an intervention node)",
        ),
        # step_4: the repair instruction names two stages as the categories to convert into.
        (
            "check if the intervention node is not better converted into a conceot node (e.g. implementation mechanism or design rationale category).",
            "check if the intervention node is not better converted into a concept node carrying one of your own category labels.",
        ),
        # step_4: the branching example is the five stages again, in prose.
        (
            "e.g. multiple problem analyses that originate from the same primary risks, multiple design rationales branching from the same theoretical insight, or multiple interventions proposed from the same validation evidence.",
            "e.g. several distinct mechanisms originating from the same risk, several approaches branching from the same claim, or several interventions proposed from the same result.",
        ),
        # output format: the category enum.
        (
            '"concept_category": "risk|problem analysis|theoretical insight|design rationale|implementation mechanism|validation evidence (concepts only, null for interventions)",',
            '"concept_category": "risk for a path-starting risk node; otherwise a short category label of your own choosing (concepts only, null for interventions)",',
        ),
    ]
    for old, new in subs:
        # Whitespace-tolerant: the released prompt carries trailing spaces on several
        # lines, and an exact-literal match would break on invisible characters rather
        # than on a real change to the instructions.
        pat = re.compile(r"\s+".join(re.escape(tok) for tok in old.split()))
        if not pat.search(out):
            fail(
                "arm-E surgery target not found in the released prompt; the prompt file "
                f"changed and this ablation would no longer be an ablation of it. "
                f"First 60 chars of the missing target: {old[:60]!r}",
                PROMPT_PY,
                "final_primary_prompt.py",
            )
        out = pat.sub(lambda _m, n=new: n, out, count=1)

    low = out.lower()
    for w in STAGE_WORDS:
        if w in low:
            fail(
                f"arm-E prompt still names the stage {w!r}",
                PROMPT_PY,
                "build_schema_blind_prompt",
            )
    for w in ["theoretical insights", "design principles", "specific mechanisms"]:
        if w in low:
            fail(
                f"arm-E prompt still paraphrases a stage as {w!r}",
                PROMPT_PY,
                "build_schema_blind_prompt",
            )
    return out


LABEL_MAP_RUBRIC = """You are mapping category labels from one extraction schema onto another.

A knowledge-graph extraction was run WITHOUT being told any category vocabulary, so it
invented its own labels for the concept nodes between a risk and a proposed intervention.
Below is the list of distinct labels it produced, each with up to three example node names.

Map each label to exactly one of these six targets:

  problem analysis      -- the mechanism by which the risk arises
  theoretical insight   -- an assumption, hypothesis or claimed resolution opportunity
  design rationale      -- the approach chosen to address the problem, and why
  implementation mechanism -- the concrete technique that realises the approach
  validation evidence   -- a measurement, result or evaluation of the approach
  unmappable            -- the label denotes something none of the five describes

Rules:
- Judge the label together with its examples, not the label string alone.
- "unmappable" is a real answer. Use it when a label denotes a role the five do not cover
  (for example background, definition, related work, or a taxonomy of the field). Do not
  stretch a label to fit; a forced mapping would manufacture the agreement this study is
  measuring.
- Return JSON only: {"mapping": [{"label": ..., "target": ..., "why": "one clause"}]}
"""


# --------------------------------------------------------------------------------------
# source degradation
# --------------------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+")


def shuffle_sentences(text: str, rng: random.Random) -> str:
    """Destroy the order of the argument, retain every sentence and all vocabulary."""
    sents = [s for s in _SENT.split(text) if s.strip()]
    rng.shuffle(sents)
    return " ".join(sents)


_REF_HEAD = re.compile(
    r"^\s{0,4}(?:\d+\s*[.)]?\s*)?(references|bibliography|works cited)\s*:?\s*$",
    re.I | re.M,
)


def references_only(text: str) -> str | None:
    """The reference list alone, or None when the document has no detectable one.

    None is reported as ineligible, never silently replaced by something else: an arm-G
    result computed over documents whose 'reference list' was actually their last section
    would measure nothing.
    """
    hits = list(_REF_HEAD.finditer(text))
    if not hits:
        return None
    tail = text[hits[-1].end() :].strip()
    if len(tail) < 500:
        return None
    return tail


# --------------------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------------------


def host_bucket(url: str) -> str:
    u = (url or "").lower()
    for host, name in [
        ("arxiv.org", "arxiv"),
        ("lesswrong.com", "lesswrong"),
        ("alignmentforum.org", "alignmentforum"),
        ("forum.effectivealtruism.org", "eaforum"),
        ("youtube.com", "youtube"),
        ("arbital.com", "arbital"),
        ("aisafety.info", "aisafety.info"),
        ("intelligence.org", "miri"),
        ("agentmodels.org", "agentmodels"),
    ]:
        if host in u:
            return name
    return "other_web"


def load_sample() -> dict:
    for path, what, how in [
        (SLIM, "slim node attributes", "experiment_review_prep_slim_nodes.py"),
        (PATHS, "the released 2,772-chain reporting unit", "phase1_dedup_paths.py"),
        (ARD, "ARD source corpus", "download_ard.py in that directory"),
    ]:
        if not path.exists():
            fail(f"{what} not found", path, how)

    na = pickle.load(open(SLIM, "rb"))
    chain_docs = {}
    with open(PATHS, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            root = rec["path"][0]
            url = na.get(root, {}).get("url")
            if url:
                chain_docs.setdefault(url, 0)
                chain_docs[url] += 1
    if not chain_docs:
        fail(
            "no chain-yielding document resolved to a URL",
            PATHS,
            "phase1_dedup_paths.py",
        )

    # ARD text for those documents, keyed by URL.
    wanted = set(chain_docs)
    texts = {}
    for fp in sorted(ARD.glob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = d.get("url")
                if u in wanted and u not in texts and (d.get("text") or "").strip():
                    texts[u] = {
                        "text": d["text"],
                        "title": d.get("title") or "",
                        "source": d.get("source") or "",
                    }

    eligible = sorted(u for u in chain_docs if u in texts)
    if len(eligible) < N_DOCS:
        fail(
            f"only {len(eligible)} chain-yielding documents have ARD source text; "
            f"{N_DOCS} are required",
            ARD,
            "download_ard.py",
        )
    rng = random.Random(SEED)
    picked = rng.sample(eligible, N_DOCS)

    # Arm G's own population: chain-yielding documents that HAVE a reference list.
    with_refs = [u for u in eligible if references_only(texts[u]["text"]) is not None]
    if len(with_refs) < N_DOCS_G:
        fail(
            f"only {len(with_refs)} chain-yielding documents carry a detectable reference "
            f"list; arm G needs {N_DOCS_G}",
            ARD,
            "download_ard.py",
        )
    picked_g = random.Random(SEED + 1).sample(with_refs, N_DOCS_G)

    return {
        "n_chain_yielding_documents": len(chain_docs),
        "n_with_ard_text": len(eligible),
        "n_with_reference_list": len(with_refs),
        "picked": picked,
        "picked_G": picked_g,
        "texts": {u: texts[u] for u in picked},
        "texts_G": {u: texts[u] for u in picked_g},
        "chains_per_doc": {u: chain_docs[u] for u in picked},
        "node_attrs": na,
    }


def doc_id(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", url)[-90:].strip("_")


# --------------------------------------------------------------------------------------
# arm-A baseline, read from the released graph
# --------------------------------------------------------------------------------------


def released_graphs(urls: list[str], na: dict) -> dict:
    if not EDGES.exists():
        fail("edge checkpoint not found", EDGES, "phase2_step1_loadandparse.py")
    keep = set(urls)
    nodes_by_url = defaultdict(dict)
    node_url = {}
    for nid, a in na.items():
        u = a.get("url")
        node_url[nid] = u
        if u in keep:
            nodes_by_url[u][nid] = a
    edges = pickle.load(open(EDGES, "rb"))
    edges_by_url = defaultdict(list)
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        u = node_url.get(e["source"])
        if u in keep:
            edges_by_url[u].append(e)
    del edges

    out = {}
    for u in urls:
        nodes = {
            nid: {
                "name": a.get("name") or str(nid),
                "type": a.get("type"),
                "category": a.get("concept_category"),
                "maturity": a.get("intervention_maturity"),
            }
            for nid, a in nodes_by_url[u].items()
        }
        eds = [
            {
                "source": e["source"],
                "target": e["target"],
                "confidence": e.get("confidence"),
            }
            for e in edges_by_url[u]
        ]
        out[u] = {"nodes": nodes, "edges": eds}
    return out


# --------------------------------------------------------------------------------------
# structural scoring: the released enumerator's rules, on one paper's graph
# --------------------------------------------------------------------------------------


def _mat(v: dict) -> int:
    """Maturity as an int; anything unparseable is 0, never assumed mature."""
    raw = v.get("maturity")
    if raw is None:
        return 0
    m = re.search(r"[1-4]", str(raw))
    return int(m.group(0)) if m else 0


def enumerate_chains(nodes: dict, edges: list) -> list:
    """Every chain the released enumerator would emit from this single-document graph.

    Same nine constraints as phase2_step4_F2v4_hopwise_falkordb.py: structural edges only
    (this graph has no similarity edges by construction), confidence >= 3 on every edge,
    endpoint maturity >= 3, exactly one risk at the root, first hop on an intermediate,
    simple paths, stop at the first qualifying intervention, 3..30 hops.
    """
    adj = defaultdict(list)
    for e in edges:
        try:
            conf = int(e.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0
        if conf < EDGE_CONFIDENCE_MIN:
            continue
        if e["source"] in nodes and e["target"] in nodes:
            adj[e["source"]].append(e["target"])

    def is_risk(n):
        return nodes[n]["type"] == "concept" and (nodes[n]["category"] or "") == "risk"

    def is_intv(n):
        return nodes[n]["type"] == "intervention"

    def mature(n):
        return _mat(nodes[n]) >= INTERVENTION_MATURITY_MIN

    chains = []
    for root in [n for n in nodes if is_risk(n)]:
        stack = [(root, [root], {root})]
        while stack:
            cur, path, seen = stack.pop()
            for nb in adj.get(cur, ()):
                if nb in seen or is_risk(nb):
                    continue
                if len(path) == 1 and is_intv(nb):
                    continue  # first hop must be an intermediate: no shortcut
                new = path + [nb]
                if is_intv(nb):
                    if mature(nb) and MIN_HOPS <= len(new) - 1 <= MAX_HOPS:
                        chains.append(new)
                    continue  # stop at the first intervention either way
                if len(new) - 1 < MAX_HOPS:
                    stack.append((nb, new, seen | {nb}))
    return chains


def score_graph(nodes: dict, edges: list, cat_map: dict | None = None) -> dict:
    """Structural profile of one document's extracted graph."""
    if cat_map:
        nodes = {
            k: {
                **v,
                "category": cat_map.get(
                    (v.get("category") or "").strip().lower(), v.get("category")
                ),
            }
            for k, v in nodes.items()
        }
    chains = enumerate_chains(nodes, edges)
    stages = Counter(
        (v.get("category") or "").strip().lower()
        for v in nodes.values()
        if v["type"] == "concept"
    )
    all5 = 0
    for c in chains:
        cats = {(nodes[n].get("category") or "").strip().lower() for n in c}
        if all(s in cats for s in STAGE_WORDS):
            all5 += 1
    return {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_edges_conf_ge3": sum(
            1 for e in edges if (e.get("confidence") or 0) and int(e["confidence"]) >= 3
        ),
        "n_risk_nodes": sum(
            1
            for v in nodes.values()
            if v["type"] == "concept" and (v.get("category") or "") == "risk"
        ),
        "n_interventions": sum(
            1 for v in nodes.values() if v["type"] == "intervention"
        ),
        "n_interventions_mature": sum(
            1
            for v in nodes.values()
            if v["type"] == "intervention" and _mat(v) >= INTERVENTION_MATURITY_MIN
        ),
        "n_chains": len(chains),
        "has_chain": bool(chains),
        "n_chains_all_five_stages": all5,
        "chain_lengths": sorted(len(c) for c in chains)[:20],
        "category_counts": dict(stages.most_common()),
    }


# --------------------------------------------------------------------------------------
# the API
# --------------------------------------------------------------------------------------

_print_lock = threading.Lock()


def openai_client():
    from dotenv import dotenv_values
    from openai import OpenAI

    if not ENV_PATH.exists():
        fail("OpenAI key file not found", ENV_PATH, "runbook R0 in paper/OPEN_ITEMS.md")
    key = (dotenv_values(ENV_PATH) or {}).get("OPENAI_API_KEY") or ""
    if not key.strip():
        fail("OPENAI_API_KEY is empty in the key file", ENV_PATH, "runbook R0")
    return OpenAI(api_key=key.strip(), timeout=TIMEOUT_S, max_retries=3)


def call_once(client, prompt: str, document: str) -> dict:
    """One request, in the exact shape extractor.py builds."""
    t0 = time.time()
    r = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_text",
                        "text": f"\n\nHere is the paper for analysis:\n\n{document}",
                    },
                ],
            }
        ],
        reasoning={"effort": REASONING_EFFORT},
    )
    usage = getattr(r, "usage", None)
    return {
        "text": r.output_text,
        "wall_clock_s": round(time.time() - t0, 1),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "reasoning_tokens": getattr(
                getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
            ),
        },
    }


_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def parse_extraction(text: str) -> dict | None:
    """Same after-the-fact parse the pipeline does: no structured-output constraint ran."""
    for cand in [m.group(1) for m in _JSON_BLOCK.finditer(text)][::-1]:
        try:
            d = json.loads(cand)
            if "nodes" in d:
                return d
        except json.JSONDecodeError:
            continue
    start = text.find('{"nodes"')
    if start < 0:
        start = text.find('{\n  "nodes"')
    if start >= 0:
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
            i += 1
    return None


def graph_from_extraction(d: dict) -> tuple[dict, list]:
    nodes = {}
    for n in d.get("nodes", []):
        name = (n.get("name") or "").strip()
        if not name:
            continue
        nodes[name] = {
            "name": name,
            "type": (n.get("type") or "concept").strip().lower(),
            "category": (n.get("concept_category") or None),
            "maturity": n.get("intervention_maturity"),
        }
    edges = []
    for e in d.get("edges", []):
        s, t = (
            (e.get("source_node") or "").strip(),
            (e.get("target_node") or "").strip(),
        )
        if s in nodes and t in nodes:
            try:
                conf = int(str(e.get("edge_confidence") or "0").strip()[:1])
            except ValueError:
                conf = 0
            edges.append({"source": s, "target": t, "confidence": conf})
    return nodes, edges


# --------------------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------------------


def build_payloads(sample: dict, prompts: dict) -> dict:
    """Arms E and F run on the common sample; arm G runs on its own.

    Only 5 of the 30 randomly drawn chain-yielding documents carry a detectable reference
    list -- forum posts, transcripts and wiki pages mostly do not have one -- so pairing
    arm G to the common sample would have measured 5 documents and called it a control.
    Arm G is therefore drawn separately from the chain-yielding documents that HAVE a
    reference list, and the receipt reports it as its own population. The manipulation is
    undefined elsewhere, and substituting a document's last section for its bibliography
    would test nothing.
    """
    rng = random.Random(SEED)
    payloads = defaultdict(dict)
    ineligible = defaultdict(list)
    for u in sample["picked"]:
        text = sample["texts"][u]["text"]
        payloads["E"][u] = (prompts["E"], text)
        payloads["F"][u] = (prompts["A"], shuffle_sentences(text, rng))
        refs = references_only(text)
        if refs is None:
            ineligible["G"].append(u)
    for u in sample["picked_G"]:
        payloads["G"][u] = (prompts["A"], references_only(sample["texts_G"][u]["text"]))
    return {"payloads": payloads, "ineligible": ineligible}


def dry_run(sample, prompts, built) -> dict:
    enc = tiktoken.get_encoding(ENCODING)
    per_arm = {}
    total_in = 0
    for arm, docs in built["payloads"].items():
        ins = [
            len(enc.encode(p, disallowed_special=()))
            + len(enc.encode(d, disallowed_special=()))
            for p, d in docs.values()
        ]
        per_arm[arm] = {
            "n_calls": len(ins),
            "input_tokens_total": sum(ins),
            "input_tokens_mean": round(statistics.mean(ins)) if ins else 0,
        }
        total_in += sum(ins)
    # Visible output: the released graph's element count for these documents, at the
    # 157 tokens/element calibration from experiment_review_extraction_cost_report.json.
    est_out_per_call = 5361
    n_calls = sum(a["n_calls"] for a in per_arm.values())
    total_out = est_out_per_call * n_calls
    band = {}
    for r in REASONING_RATIOS:
        billed_out = total_out * (1 + r)
        band[f"reasoning_{r:g}x"] = round(
            total_in / 1e6 * RATE_IN + billed_out / 1e6 * RATE_OUT, 2
        )
    return {
        "per_arm": per_arm,
        "n_calls": n_calls,
        "input_tokens_total": total_in,
        "visible_output_tokens_assumed": total_out,
        "visible_output_per_call_assumed": est_out_per_call,
        "usd_band_by_reasoning_ratio": band,
        "wall_clock_estimate_min": round(n_calls * 90 / max(1, MAX_WORKERS) / 60, 1),
        "rates_usd_per_million_ASSUMED": {"input": RATE_IN, "output": RATE_OUT},
        "ineligible": {k: len(v) for k, v in built["ineligible"].items()},
    }


def run(arms: list[str], sample, built, limit: int = 0) -> None:
    client = openai_client()
    jobs = []
    for arm in arms:
        (RAW_DIR / arm).mkdir(parents=True, exist_ok=True)
        for u, (prompt, doc) in built["payloads"][arm].items():
            dest = RAW_DIR / arm / f"{doc_id(u)}.json"
            if dest.exists():
                continue
            jobs.append((arm, u, prompt, doc, dest))
    if limit:
        jobs = jobs[:limit]  # smoke run: one call, checked end to end before the rest
    print(f"{len(jobs)} calls to make ({MAX_WORKERS} at a time)", flush=True)

    def work(job):
        arm, u, prompt, doc, dest = job
        res = call_once(client, prompt, doc)
        dest.write_text(
            json.dumps(
                {
                    "arm": arm,
                    "url": u,
                    "model": MODEL,
                    "effort": REASONING_EFFORT,
                    **res,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return arm, u, res

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            arm, u, _, _, _ = futs[f]
            done += 1
            try:
                _, _, res = f.result()
                msg = (
                    f"[{done}/{len(jobs)}] {arm} {doc_id(u)[:40]} "
                    f"{res['wall_clock_s']}s in={res['usage']['input_tokens']} "
                    f"out={res['usage']['output_tokens']} "
                    f"reasoning={res['usage']['reasoning_tokens']}"
                )
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                msg = f"[{done}/{len(jobs)}] {arm} {doc_id(u)[:40]} FAILED {type(exc).__name__}: {exc}"
            with _print_lock:
                print(msg, flush=True)


def map_labels(client, labels: dict) -> dict:
    """One call: free-form arm-E labels onto the five stages, by the stated rubric."""
    listing = "\n".join(
        f"- {lab}  (examples: {'; '.join(ex[:3])})"
        for lab, ex in sorted(labels.items())
    )
    r = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": LABEL_MAP_RUBRIC},
                    {"type": "input_text", "text": listing},
                ],
            }
        ],
        reasoning={"effort": REASONING_EFFORT},
    )
    d = parse_extraction(r.output_text) or {}
    if "mapping" not in d:
        m = re.search(r'\{.*"mapping".*\}', r.output_text, re.S)
        d = json.loads(m.group(0)) if m else {"mapping": []}
    return {
        "raw": r.output_text,
        "map": {
            row["label"].strip().lower(): row["target"].strip().lower()
            for row in d.get("mapping", [])
            if row.get("label") and row.get("target")
        },
    }


def score(sample, built) -> dict:
    all_urls = list(dict.fromkeys(sample["picked"] + sample["picked_G"]))
    base = released_graphs(all_urls, sample["node_attrs"])
    arms = {}
    arm_a = {}
    for u in all_urls:
        g = base[u]
        arm_a[u] = score_graph(g["nodes"], g["edges"])
    arms["A_released"] = arm_a

    parsed = {}
    for arm in ["E", "F", "G"]:
        d = RAW_DIR / arm
        if not d.exists():
            continue
        got = {}
        for fp in sorted(d.glob("*.json")):
            rec = json.loads(fp.read_text(encoding="utf-8"))
            ext = parse_extraction(rec.get("text") or "")
            got[rec["url"]] = {"rec": rec, "ext": ext}
        parsed[arm] = got

    # arm E label mapping
    cat_map, label_receipt = {}, None
    if "E" in parsed:
        labels = defaultdict(list)
        for u, p in parsed["E"].items():
            if not p["ext"]:
                continue
            for n in p["ext"].get("nodes", []):
                c = (n.get("concept_category") or "").strip().lower()
                if c and c != "risk" and (n.get("type") or "concept") == "concept":
                    labels[c].append((n.get("name") or "")[:90])
        if labels:
            client = openai_client()
            label_receipt = map_labels(client, labels)
            cat_map = label_receipt["map"]
            cat_map["risk"] = "risk"

    for arm, got in parsed.items():
        rows = {}
        for u, p in got.items():
            if not p["ext"]:
                rows[u] = {"parse_failure": True}
                continue
            nodes, edges = graph_from_extraction(p["ext"])
            rows[u] = score_graph(nodes, edges, cat_map if arm == "E" else None)
            rows[u]["usage"] = p["rec"].get("usage")
        arms[arm] = rows

    def agg(rows):
        ok = [r for r in rows.values() if not r.get("parse_failure")]
        if not ok:
            return {"n": 0}
        return {
            "n": len(ok),
            "parse_failures": sum(1 for r in rows.values() if r.get("parse_failure")),
            "pct_with_chain": round(100 * sum(r["has_chain"] for r in ok) / len(ok), 1),
            "mean_chains": round(statistics.mean([r["n_chains"] for r in ok]), 2),
            "mean_nodes": round(statistics.mean([r["n_nodes"] for r in ok]), 1),
            "mean_edges": round(statistics.mean([r["n_edges"] for r in ok]), 1),
            "pct_chains_all_five": (
                round(
                    100
                    * sum(r["n_chains_all_five_stages"] for r in ok)
                    / max(1, sum(r["n_chains"] for r in ok)),
                    1,
                )
            ),
        }

    return {
        "headline": {arm: agg(rows) for arm, rows in arms.items()},
        "per_document": arms,
        "arm_E_label_mapping": label_receipt,
        "arm_E_rubric": LABEL_MAP_RUBRIC,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", default="", help="comma-separated arms to execute, e.g. E,F,G"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="smoke run: stop after N calls"
    )
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    released = load_released_prompt()
    prompts = {"A": released, "E": build_schema_blind_prompt(released)}
    sample = load_sample()
    built = build_payloads(sample, prompts)

    receipt = {
        "study": "schema ablation and degraded-source control (issue #165, S6+S8, R2)",
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "transport": "synchronous /v1/responses; the corpus run used the batch API",
        "sample": {
            "n_documents": len(sample["picked"]),
            "seed": SEED,
            "drawn_from": "chain-yielding documents with ARD source text",
            "n_chain_yielding_documents": sample["n_chain_yielding_documents"],
            "n_with_ard_text": sample["n_with_ard_text"],
            "source_mix": dict(
                Counter(host_bucket(u) for u in sample["picked"]).most_common()
            ),
            "urls": sample["picked"],
        },
        "arm_G_population": {
            "n_documents": len(sample["picked_G"]),
            "seed": SEED + 1,
            "drawn_from": "chain-yielding documents carrying a detectable reference list",
            "n_eligible": sample["n_with_reference_list"],
            "urls": sample["picked_G"],
            "why_separate": (
                "only "
                f"{N_DOCS - len(built['ineligible']['G'])} of the {N_DOCS} documents in the "
                "common sample carry a reference list, so pairing arm G to it would have "
                "measured five documents"
            ),
        },
        "arm_G_ineligible_in_common_sample": built["ineligible"]["G"],
    }

    if args.run:
        run(
            [a.strip().upper() for a in args.run.split(",") if a.strip()],
            sample,
            built,
            args.limit,
        )
    if args.score:
        receipt["results"] = score(sample, built)
    if not args.run and not args.score:
        receipt["dry_run"] = dry_run(sample, prompts, built)
        print(json.dumps(receipt["dry_run"], indent=2))
        print(
            f"\narm-E prompt: {len(prompts['E'])} chars vs released {len(prompts['A'])}"
        )
        print(
            f"arm-G ineligible (no reference list): {len(built['ineligible']['G'])} of {N_DOCS}"
        )

    receipt["wall_clock_s"] = round(time.time() - t0, 1)
    if args.score:
        OUT.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote {OUT}")
        print(json.dumps(receipt["results"]["headline"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
