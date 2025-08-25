from pathlib import Path
import json
from falkordb import FalkorDB
from tqdm import tqdm

try:
    from intervention_graph_creation.src.prompts import OutputSchema
except ImportError:
    from prompts import OutputSchema

HOST = "localhost"
PORT = 6379
GRAPH = "AISafetyIntervention"


def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


class AISafetyGraph:
    """Minimal wrapper for FalkorDB ingestion for AISafetyIntervention graph."""

    def __init__(self) -> None:
        self.db = FalkorDB(host=HOST, port=PORT)

    def upsert_paper(self, paper_id: str) -> None:
        g = self.db.select_graph(GRAPH)
        g.query(f"MERGE (p:PAPER {{id: '{paper_id}'}}) RETURN p")

    def upsert_edge(self, paper_id: str, edge) -> None:
        g = self.db.select_graph(GRAPH)
        n = edge.target_node
        # assumes that two nodes having the same name are the same node
        # (could be problematic as there are multiple nodes with the same name under different contexts)
        g.query(
            f"MERGE (t:{n.type} {{name: {lit(n.name)}}}) "
            f"SET t.canonical_name = {lit(n.canonical_name)}, "
            f"t.aliases = {lit(n.aliases)}, "
            f"t.confidence = {lit(n.confidence)}, "
            f"t.notes = {lit(n.notes)} "
            f"RETURN t"
        )
        # We'll be pushing a single paper schema and merging them?
        g.query(
            f"MATCH (p:PAPER {{id: '{paper_id}'}}), (t:{n.type} {{name: {lit(n.name)}}}) "
            f"MERGE (p)-[r:{edge.type}]->(t) "
            f"SET r.rationale = {lit(edge.rationale)}, r.confidence = {lit(edge.confidence)} "
            f"RETURN 1"
        )

    def ingest_dir(self, output_dir: str = "output") -> None:
        paths = Path("").glob(f"{output_dir}/*.json")
        paths = [x for x in paths if x.is_file() and "raw_response" not in x.name]
        for json_path in tqdm(paths):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            paper_id = json_path.stem
            self.upsert_paper(paper_id)
            for edge in doc.edges:
                self.upsert_edge(paper_id, edge)

    def get_nodes(self) -> list[dict]:
        g = self.db.select_graph(GRAPH)
        res = g.ro_query(
            "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, n AS node "
        )

        out = []
        for row in res.result_set:
            node_id = row[0]
            labels = row[1] or []
            node = row[2]
            props = node.properties or {}

            parts = []
            for k in props.keys():
                if k in ("confidence", "id"):
                    continue
                v = props[k]
                if isinstance(v, str):
                    v_str = v
                elif isinstance(v, (list, tuple)):
                    v_str = ", ".join(str(x) for x in v)
                else:
                    v_str = str(v)
                if len(v_str) > 0:
                    parts.append(f"{k}={v_str}")

            text = "; ".join(parts) if parts else ""
            if text:
                out.append(
                    {
                        "id": node_id,
                        "labels": labels,
                        "text": text,
                    }
                )
        return out

    def merge_nodes(self, keep_id: int, remove_id: int):
        """
        TODO: use FalkorDB vector index to find ids
        """

        q = f"""
        MATCH (n) WHERE ID(n) = {remove_id}
        OPTIONAL MATCH (n)-[r]->() RETURN DISTINCT type(r) AS t
        UNION
        MATCH (n) WHERE ID(n) = {remove_id}
        OPTIONAL MATCH ()-[r]->(n) RETURN DISTINCT type(r) AS t
        """
        graph = self.db.select_graph(GRAPH)
        result = graph.query(q)
        rel_types = [r[0] for r in result.result_set if r[0] is not None]

        if not rel_types:
            graph.query(f"MATCH (a) WHERE ID(a) = {remove_id} DELETE a")
            return

        parts = []
        for rtype in rel_types:
            parts.append(f"""
            // outgoing {rtype}
            OPTIONAL MATCH (a)-[r:{rtype}]->(m)
            WITH a, b, r, m
            FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
                MERGE (b)-[r2:{rtype}]->(m)
                SET r2 += r
            )
            WITH a, b

            // incoming {rtype}
            OPTIONAL MATCH (m2)-[s:{rtype}]->(a)
            WITH a, b, s, m2
            FOREACH (_ IN CASE WHEN m2 IS NULL THEN [] ELSE [1] END |
                MERGE (m2)-[s2:{rtype}]->(b)
                SET s2 += s
            )
            WITH a, b
            """)

        merge_query = f"""
        MATCH (a), (b)
        WHERE ID(a) = {remove_id} AND ID(b) = {keep_id}
        {"".join(parts)}
        DELETE a
        """

        return graph.query(merge_query)


def main():
    AISafetyGraph().ingest_dir("output")


if __name__ == "__main__":
    main()
