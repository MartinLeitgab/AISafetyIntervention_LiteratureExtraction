"""
Lightweight PreLLM KG/JSON validator

Validates:
- Schema shape and required fields
- ID format and uniqueness
- Referential integrity (edges -> nodes, chains alternate node/edge IDs & all IDs exist)
- Orphan nodes (degree 0 and unused in chains)
- Duplicate nodes by (type, normalized name)
"""

import json
import sys
import re
from typing import Dict, Any, List, Set

ALLOWED_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def _is_str(x):
    return isinstance(x, str)


def _is_dict(x):
    return isinstance(x, dict)


def _is_list(x):
    return isinstance(x, list)


def ok(value: bool) -> str:
    return "PASS" if value else "FAIL"
async def upload_and_create_batch(batch_file_name: str) -> str:

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    #Upload the file
    with open(batch_file_name, "rb") as f:
        uploaded_file = await client.files.create(
            file=f,
            purpose="batch"
        )

    # Create the batch
    batch = await client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch.id

def validate_schema(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    # Top-level
    for key, typ in [
        ("nodes", list),
        ("edges", list),
        ("chains", list),
        ("meta", dict),
    ]:
        if key not in data:
            issues.append(
                {
                    "severity": "BLOCKER",
                    "issue": f"Missing top-level key '{key}'",
                    "where": key,
                }
            )
        else:
            if not isinstance(data[key], typ):
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "issue": f"Key '{key}' must be {typ.__name__}",
                        "where": key,
                    }
                )
    if "meta" in data and isinstance(data["meta"], dict):
        meta = data["meta"]
        if "version" in meta and not isinstance(meta["version"], str):
            issues.append(
                {
                    "severity": "MINOR",
                    "issue": "meta.version should be string",
                    "where": "meta.version",
                }
            )
        if "source_hash" in meta and not isinstance(meta["source_hash"], str):
            issues.append(
                {
                    "severity": "MINOR",
                    "issue": "meta.source_hash should be string",
                    "where": "meta.source_hash",
                }
            )

    # Nodes
    for i, n in enumerate(data.get("nodes", [])):
        path = f"nodes[{i}]"
        if not _is_dict(n):
            issues.append(
                {"severity": "BLOCKER", "issue": "Node must be object", "where": path}
            )
            continue
        for req in ["id", "type", "name"]:
            if req not in n or not isinstance(n[req], str) or not n[req].strip():
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "issue": f"Node missing/invalid '{req}'",
                        "where": f"{path}.{req}",
                    }
                )
        if "props" in n and not _is_dict(n["props"]):
            issues.append(
                {
                    "severity": "MINOR",
                    "issue": "Node.props should be object",
                    "where": f"{path}.props",
                }
            )

    # Edges
    for i, e in enumerate(data.get("edges", [])):
        path = f"edges[{i}]"
        if not _is_dict(e):
            issues.append(
                {"severity": "BLOCKER", "issue": "Edge must be object", "where": path}
            )
            continue
        for req in ["id", "type", "source", "target"]:
            if req not in e or not isinstance(e[req], str) or not e[req].strip():
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "issue": f"Edge missing/invalid '{req}'",
                        "where": f"{path}.{req}",
                    }
                )
        if "props" in e and not _is_dict(e["props"]):
            issues.append(
                {
                    "severity": "MINOR",
                    "issue": "Edge.props should be object",
                    "where": f"{path}.props",
                }
            )


    return issues


def validate_id_format_and_uniqueness(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []

    def _check_unique(items, kind: str):
        seen: Set[str] = set()
        for i, obj in enumerate(items):
            _id = obj.get("id")
            path = f"{kind}[{i}].id"
            if not isinstance(_id, str) or not _id:
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "issue": f"{kind[:-1].capitalize()} id must be non-empty string",
                        "where": path,
                    }
                )
                continue
            if not ALLOWED_ID_RE.match(_id):
                issues.append(
                    {
                        "severity": "MAJOR",
                        "issue": f"Invalid id format '{_id}'",
                        "where": path,
                        "suggestion": "Use [A-Za-z0-9._:-]",
                    }
                )
            if _id in seen:
                issues.append(
                    {"severity": "BLOCKER", "issue": "Duplicate id", "where": path}
                )
            seen.add(_id)

    _check_unique(data.get("nodes", []), "nodes")
    _check_unique(data.get("edges", []), "edges")
    _check_unique(data.get("chains", []), "chains")
    return issues


def validate_referential_integrity(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    node_ids = {n.get("id") for n in data.get("nodes", []) if isinstance(n, dict)}
    edge_ids = {e.get("id") for e in data.get("edges", []) if isinstance(e, dict)}

    # Edge endpoints exist
    for i, e in enumerate(data.get("edges", [])):
        src, tgt = e.get("source"), e.get("target")
        if src not in node_ids:
            issues.append(
                {
                    "severity": "BLOCKER",
                    "issue": "Edge source does not exist",
                    "where": f"edges[{i}].source",
                    "ids": [e.get("id"), src],
                }
            )
        if tgt not in node_ids:
            issues.append(
                {
                    "severity": "BLOCKER",
                    "issue": "Edge target does not exist",
                    "where": f"edges[{i}].target",
                    "ids": [e.get("id"), tgt],
                }
            )

    # Chains: steps alternate node, edge, node ... and refer to existing ids
    for i, c in enumerate(data.get("chains", [])):
        steps = c.get("steps", [])
        if not steps:
            continue
        # Identify expected parity: even index -> node, odd -> edge
        bad_ref = False
        for j, sid in enumerate(steps):
            if not isinstance(sid, str):
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "issue": "Chain step ID must be string",
                        "where": f"chains[{i}].steps[{j}]",
                    }
                )
                bad_ref = True
                continue
            if j % 2 == 0:
                if sid not in node_ids:
                    issues.append(
                        {
                            "severity": "BLOCKER",
                            "issue": "Chain expects node id at even index",
                            "where": f"chains[{i}].steps[{j}]",
                            "ids": [sid],
                        }
                    )
                    bad_ref = True
            else:
                if sid not in edge_ids:
                    issues.append(
                        {
                            "severity": "BLOCKER",
                            "issue": "Chain expects edge id at odd index",
                            "where": f"chains[{i}].steps[{j}]",
                            "ids": [sid],
                        }
                    )
                    bad_ref = True
        if steps and len(steps) % 2 == 0:
            issues.append(
                {
                    "severity": "BLOCKER",
                    "issue": "Chain must start and end with a node id (odd number of steps)",
                    "where": f"chains[{i}].steps",
                }
            )
        if bad_ref:
            continue

    # Orphans: nodes not referenced by any edge or chain
    node_degree = {nid: 0 for nid in node_ids}
    for e in data.get("edges", []):
        if e.get("source") in node_degree:
            node_degree[e.get("source")] += 1
        if e.get("target") in node_degree:
            node_degree[e.get("target")] += 1
    for c in data.get("chains", []):
        for j, sid in enumerate(c.get("steps", [])):
            if j % 2 == 0 and sid in node_degree:
                node_degree[sid] += 1
    for nid, deg in node_degree.items():
        if deg == 0:
            issues.append(
                {
                    "severity": "MAJOR",
                    "issue": "Orphan node (degree 0 & unused in chains)",
                    "where": f"nodes[id={nid}]",
                    "ids": [nid],
                }
            )

    return issues


def detect_duplicate_nodes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    seen = {}
    for i, n in enumerate(data.get("nodes", [])):
        key = (str(n.get("type")).strip().lower(), str(n.get("name")).strip().lower())
        if key in seen:
            issues.append(
                {
                    "severity": "MINOR",
                    "issue": "Potential duplicate nodes (same type+name)",
                    "where": f"nodes[{i}]",
                    "ids": [seen[key], n.get("id")],
                    "suggestion": "Merge one into the other and retarget edges.",
                }
            )
        else:
            seen[key] = n.get("id")
    return issues


def validate_all(data: Dict[str, Any]) -> Dict[str, Any]:
    schema_issues = validate_schema(data)
    id_issues = validate_id_format_and_uniqueness(data)
    ref_issues = (
        validate_referential_integrity(data)
        if not any(i["severity"] == "BLOCKER" for i in schema_issues + id_issues)
        else []
    )
    dup_issues = (
        detect_duplicate_nodes(data)
        if not any(i["severity"] == "BLOCKER" for i in schema_issues + id_issues)
        else []
    )

    has_blockers = any(
        i["severity"] == "BLOCKER" for i in (schema_issues + id_issues + ref_issues)
    )
    report = {
        "decision": {
            "is_valid_json": True,
            "has_blockers": has_blockers,
            "summary": "Valid KG JSON"
            if not has_blockers
            else "KG JSON has blocking issues; see validation_report",
        },
        "validation_report": {
            "schema_check": schema_issues,
            "id_check": id_issues,
            "referential_check": ref_issues,
            "duplicates": dup_issues,
        },
    }
    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python kg_validator.py <input.json>")
        sys.exit(2)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    report = validate_all(data)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
