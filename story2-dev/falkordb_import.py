"""
FalkorDB graph importer: nodes, edges, UUIDs, alias normalization, and index setup.

Usage (as a library):
    from falkordb_import_refactored import FalkorImporter
    importer = FalkorImporter(host="localhost", port=6379, graph="test")
    ok, info = importer.ingest(data_or_path="path/to.json")  # or pass a dict via data_or_path=dict(...)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from falkordb import FalkorDB


def _dedup_preserve_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def normalize_aliases(val: Any) -> List[str]:
    """
    Normalize aliases into a clean list[str], handling commas inside list elements.
    Accepts:
      - str: "a, b, c"
      - list[str]: ["a, b", "c"]  (elements may also be comma-separated)
    Returns a lowercased, trimmed, de-duplicated list in stable order.
    """
    tokens: List[str] = []

    def add_from_string(s: str) -> None:
        for part in s.split(","):
            p = part.strip().lower()
            if p:
                tokens.append(p)

    if val is None:
        return []
    if isinstance(val, str):
        add_from_string(val)
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, str):
                add_from_string(item)
    # ignore other types silently
    return _dedup_preserve_order(tokens)


def sanitize_props(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a dict of properties:
      - drop None values
      - drop 'type' (label encodes type)
      - lower-case all string values (except 'aliases' which is handled specially)
      - convert 'aliases' into a list[str], supporting comma-separated inputs
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "type" or v is None:
            continue
        if k == "aliases":
            aliases = normalize_aliases(v)
            if aliases:
                out["aliases"] = aliases
            continue
        out[k] = v.lower() if isinstance(v, str) else v
    return out


@dataclass
class IngestStats:
    nodes_before: int = 0
    edges_before: int = 0
    nodes_after: int = 0
    edges_after: int = 0
    edges_created: int = 0
    edges_skipped: int = 0


class FalkorImporter:
    """
    Importer that ingests nodes and edges into a FalkorDB graph.

    Features:
      - Creates nodes with labels 'concept' or 'intervention'
      - Assigns a per-node UUID in 'uid'
      - Preserves aliases as a list[str]
      - Resolves edges by JSON 'id' first, otherwise by unique 'name'
      - Creates an index on :concept(concept_category) (idempotent)
      - Returns (success, info) tuples; info contains stats or error/traceback.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph: str = "test",
        password: Optional[str] = None,
    ) -> None:
        self._db = FalkorDB(host=host, port=port, password=password)
        self._g = self._db.select_graph(graph)

    # ---------- public API ----------

    def ingest(
        self,
        data_or_path: Union[str, Dict[str, Any]],
        create_index: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Ingest a JSON structure (dict) or load from a JSON file path.
        Returns (success, info) where:
          - success: bool
          - info: on success -> {'stats': IngestStats-as-dict}
                  on failure -> {'error': str, 'traceback': str}
        """
        try:
            data = self._load_json(data_or_path)
            stats = IngestStats()

            stats.nodes_before = self._count_nodes()
            stats.edges_before = self._count_edges()

            id_to_uid, name_to_uids = self._pass_create_nodes(data)
            created, skipped = self._pass_create_edges(data, id_to_uid, name_to_uids)
            stats.edges_created = created
            stats.edges_skipped = skipped

            if create_index:
                self._create_indexes()

            stats.nodes_after = self._count_nodes()
            stats.edges_after = self._count_edges()

            return True, {"stats": vars(stats)}
        except Exception as e:
            return False, {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

    # ---------- internal helpers ----------

    def _load_json(self, data_or_path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(data_or_path, dict):
            return data_or_path
        # assume path
        with open(str(data_or_path), "r", encoding="utf-8") as f:
            return json.load(f)

    def _count_nodes(self) -> int:
        res = self._g.query("MATCH (n) RETURN count(n) AS cnt")
        return res.result_set[0][0] if res.result_set else 0

    def _count_edges(self) -> int:
        res = self._g.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return res.result_set[0][0] if res.result_set else 0

    def _pass_create_nodes(self, data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        nodes = data.get("nodes", []) or []
        id_to_uid: Dict[str, str] = {}
        name_to_uids: Dict[str, List[str]] = {}

        for item in nodes:
            node_type = item.get("type")
            if node_type not in {"concept", "intervention"}:
                continue

            label = node_type  # keep exact lower-case labels
            props = sanitize_props(item)
            uid = str(uuid.uuid4())
            props["uid"] = uid

            if item.get("id") is not None:
                id_to_uid[item["id"]] = uid
            if item.get("name") is not None:
                name_to_uids.setdefault(item["name"], []).append(uid)

            q = f"""
            CREATE (n:`{label}`)
            SET n = $props
            RETURN n
            """
            self._g.query(q, {"props": props})

        return id_to_uid, name_to_uids

    def _resolve_ref(self, ref_value: Optional[str], id_to_uid: Dict[str, str], name_to_uids: Dict[str, List[str]]) -> Optional[str]:
        if ref_value is None:
            return None
        if ref_value in id_to_uid:
            return id_to_uid[ref_value]
        uids = name_to_uids.get(ref_value, [])
        if len(uids) == 1:
            return uids[0]
        return None  # ambiguous or not found

    def _pass_create_edges(self, data: Dict[str, Any], id_to_uid: Dict[str, str], name_to_uids: Dict[str, List[str]]) -> Tuple[int, int]:
        chains = data.get("logical_chains", []) or []
        made_edges = 0
        skipped_edges = 0

        for chain in chains:
            for e in chain.get("edges", []) or []:
                rel_type = e.get("type")
                src_ref = e.get("source_node")
                dst_ref = e.get("target_node")
                desc = e.get("description")
                conf = e.get("edge_confidence")

                if not (rel_type and src_ref is not None and dst_ref is not None):
                    skipped_edges += 1
                    continue

                src_uid = self._resolve_ref(src_ref, id_to_uid, name_to_uids)
                dst_uid = self._resolve_ref(dst_ref, id_to_uid, name_to_uids)
                if not (src_uid and dst_uid):
                    skipped_edges += 1
                    continue

                q = f"""
                MATCH (s {{uid: $src_uid}}), (t {{uid: $dst_uid}})
                CREATE (s)-[r:`{rel_type}`]->(t)
                SET r.description = $desc, r.edge_confidence = $conf
                RETURN s, r, t
                """
                self._g.query(q, {"src_uid": src_uid, "dst_uid": dst_uid, "desc": desc, "conf": conf})
                made_edges += 1

        return made_edges, skipped_edges

    def _create_indexes(self) -> None:
        # exact-match index
        try:
            self._g.query("CREATE INDEX ON :concept(concept_category)")
        except Exception:
            # Likely already exists; ignore
            pass
        # optional: full-text on name + aliases (commented out; enable if RediSearch is configured)
        # try:
        #     self._g.query("CALL db.idx.fulltext.createNodeIndex('concept', 'name', 'aliases')")
        # except Exception:
        #     pass



if __name__ == "__main__":
    sys.exit()
