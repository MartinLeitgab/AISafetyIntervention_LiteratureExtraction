from pathlib import Path
import logging
import json
from falkordb import FalkorDB
from tqdm import tqdm
from typing import List

from config import load_settings
from intervention_graph_creation.src.local_graph_extraction.core import Node, Edge, PaperSchema
from intervention_graph_creation.src.local_graph_extraction.db.helpers import label_for, lit


SETTINGS = load_settings()
logger = logging.getLogger(__name__)


class AISafetyGraph:
    def __init__(self) -> None:
        self.db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)

    # ---------- nodes ----------

    def upsert_node(self, node: Node, meta_kv: dict) -> None:
        g = self.db.select_graph(SETTINGS.falkordb.graph)
        label = label_for(node.type)
        # Uniqueness by (name, type) → prevents duplicates for same typed name
        set_parts = [
            f"n.description = {lit(node.description)}",
            f"n.aliases = {lit(node.aliases)}",
            f"n.concept_category = {lit(node.concept_category)}",
            f"n.intervention_lifecycle = {lit(node.intervention_lifecycle)}",
            f"n.intervention_maturity = {lit(node.intervention_maturity)}",
        ]
        for k, v in (meta_kv or {}).items():
            if k and v is not None:
                set_parts.append(f"n.{k} = {lit(v)}")
        set_clause = ", ".join(set_parts)
        g.query(
            f"MERGE (n:{label} {{name: {lit(node.name)}, type: {lit(node.type)}}}) "
            f"SET {set_clause} "
            f"RETURN n"
        )

    # ---------- edges ----------
    # Multiple edges between same nodes are allowed,
    # but for the same etype we update the existing edge (MERGE by etype).

    def upsert_edge(self, edge: Edge, meta_kv: dict) -> None:
        g = self.db.select_graph(SETTINGS.falkordb.graph)
        s = lit(edge.source_node)
        t = lit(edge.target_node)
        etype = lit(edge.type)

        # Ensure endpoints exist (by name only; labels may be added elsewhere)
        g.query(f"MERGE (a {{name: {s}}}) RETURN a")
        g.query(f"MERGE (b {{name: {t}}}) RETURN b")

        # One :EDGE per (a,b,etype). If exists → update props; else → create.
        set_parts = [
            "r.description = " + lit(edge.description),
            "r.edge_confidence = " + lit(edge.edge_confidence),
        ]
        for k, v in (meta_kv or {}).items():
            if k and v is not None:
                set_parts.append("r." + k + " = " + lit(v))
        set_clause = ", ".join(set_parts)
        g.query(
            "MATCH (a {name: " + s + "}), (b {name: " + t + "}) "
            "MERGE (a)-[r:EDGE {etype: " + etype + "}]->(b) "
            "SET " + set_clause + " "
            "RETURN r"
        )

    # ---------- ingest ----------

    def ingest_file(self, json_path: Path) -> None:
        try:
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
            doc = PaperSchema(**data)
        except Exception as e:
            logger.exception("Failed to parse or validate JSON file: %s", json_path)
            raise

        # Basic file-level checks
        names = [n.name for n in doc.nodes]
        if len(names) != len(set(names)):
            dupes = sorted({x for x in names if names.count(x) > 1})
            raise ValueError(f"Duplicate node names in {json_path.name}: {dupes}")

        known = set(names)
        missing = [
            (e.source_node, e.target_node)
            for ch in doc.logical_chains
            for e in ch.edges
            if e.source_node not in known or e.target_node not in known
        ]
        if missing:
            raise ValueError(f"Edges reference unknown nodes in {json_path.name}: {missing[:5]}...")

        # Serialize paper-level metadata once; attach to all nodes/edges
        try:
            meta_kv = {m.key: m.value for m in (doc.meta or [])}
        except Exception:
            logger.exception("Failed to construct meta_kv for %s; defaulting to empty metadata", json_path.name)
            meta_kv = {}

        for n in doc.nodes:
            try:
                self.upsert_node(n, meta_kv)
            except Exception:
                logger.exception("Node upsert failed for '%s' (type=%s) in %s", n.name, n.type, json_path.name)
                continue

        for ch in doc.logical_chains:
            for e in ch.edges:
                try:
                    self.upsert_edge(e, meta_kv)
                except Exception:
                    logger.exception(
                        "Edge upsert failed (%s: %s -> %s) in %s",
                        e.type,
                        e.source_node,
                        e.target_node,
                        json_path.name,
                    )
                    continue

    def ingest_dir(self, input_dir: Path = SETTINGS.paths.output_dir) -> None:
        base = Path(input_dir)
        subdirs = [d for d in base.iterdir() if d.is_dir()]
        for d in tqdm(sorted(subdirs)):
            json_path = d / f"{d.name}.json"
            if not json_path.exists():
                logger.warning("Skipping %s: %s not found", d.name, json_path)
                continue
            try:
                self.ingest_file(json_path)
            except Exception:
                logger.exception("Ingest failed for %s", json_path)
                continue

    # ---------- utils ----------

    def get_nodes(self) -> List[dict]:
        g = self.db.select_graph(SETTINGS.falkordb.graph)
        res = g.ro_query("MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, n AS node")
        out = []
        for row in res.result_set:
            node_id = row[0]
            labels = row[1] or []
            node = row[2]
            props = node.properties or {}
            parts = []
            for k, v in props.items():
                if k in ("id",):
                    continue
                if isinstance(v, str):
                    v_str = v
                elif isinstance(v, (list, tuple)):
                    v_str = ", ".join(str(x) for x in v)
                else:
                    v_str = str(v)
                if v_str:
                    parts.append(f"{k}={v_str}")
            text = "; ".join(parts) if parts else ""
            if text:
                out.append({"id": node_id, "labels": labels, "text": text})
        return out

    def merge_nodes(self, keep_name: str, remove_name: str):
        """
        Merge two nodes identified by name.
        Moves all relationships from remove_name -> keep_name, then deletes remove_name.
        """
        graph = self.db.select_graph(SETTINGS.falkordb.graph)

        q = f"""
        MATCH (n {{name: {lit(remove_name)}}})
        OPTIONAL MATCH (n)-[r]->() RETURN DISTINCT type(r) AS t
        UNION
        MATCH (n {{name: {lit(remove_name)}}})
        OPTIONAL MATCH ()-[r]->(n) RETURN DISTINCT type(r) AS t
        """
        result = graph.query(q)
        rel_types = [r[0] for r in result.result_set if r[0] is not None]

        if not rel_types:
            return graph.query(f"MATCH (a {{name: {lit(remove_name)}}}) DELETE a")

        parts = []
        for rtype in rel_types:
            parts.append(f"""
            OPTIONAL MATCH (a {{name: {lit(remove_name)}}})-[r:{rtype}]->(m)
            MATCH (b {{name: {lit(keep_name)}}})
            FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
                MERGE (b)-[r2:{rtype}]->(m)
                SET r2 += r
            )
            WITH a, b
            OPTIONAL MATCH (m2)-[s:{rtype}]->(a {{name: {lit(remove_name)}}})
            FOREACH (_ IN CASE WHEN m2 IS NULL THEN [] ELSE [1] END |
                MERGE (m2)-[s2:{rtype}]->(b)
                SET s2 += s
            )
            WITH a, b
            """)

        merge_query = f"""
        MATCH (a {{name: {lit(remove_name)}}}), (b {{name: {lit(keep_name)}}})
        {"".join(parts)}
        DELETE a
        """
        return graph.query(merge_query)


def main():
    AISafetyGraph().ingest_dir(SETTINGS.paths.output_dir)


if __name__ == "__main__":
    main()
