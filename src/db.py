from pathlib import Path
import json
from falkordb import FalkorDB
from tqdm import tqdm
try:
    from src.prompts import OutputSchema
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


def main():
    paths = Path(".").glob("output/*.json")
    paths = [x for x in paths if x.is_file() and "raw_response" not in x.name]

    for json_path in tqdm(paths):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc = OutputSchema(**data)
        paper_id = json_path.stem
 
        db = FalkorDB(host=HOST, port=PORT)
        g = db.select_graph(GRAPH)
        g.query(f"MERGE (p:PAPER {{id: '{paper_id}'}}) RETURN p")
 
        for edge in doc.edges:
            n = edge.target_node
            g.query(
                f"MERGE (t:{n.type} {{name: {lit(n.name)}}}) "
                f"SET t.canonical_name = {lit(n.canonical_name)}, "
                f"t.aliases = {lit(n.aliases)}, "
                f"t.confidence = {lit(n.confidence)}, "
                f"t.notes = {lit(n.notes)} "
                f"RETURN t"
            )
            g.query(
                f"MATCH (p:PAPER {{id: '{paper_id}'}}), (t:{n.type} {{name: {lit(n.name)}}}) "
                f"MERGE (p)-[r:{edge.type}]->(t) "
                f"SET r.rationale = {lit(edge.rationale)}, r.confidence = {lit(edge.confidence)} "
                f"RETURN 1"
            )


if __name__ == "__main__":
    main()
