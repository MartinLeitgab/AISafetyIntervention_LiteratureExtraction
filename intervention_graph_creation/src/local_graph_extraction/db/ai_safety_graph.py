from pathlib import Path
import json
import shutil
from falkordb import FalkorDB
from tqdm import tqdm
from typing import List, Dict, Any

from config import load_settings
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import PaperSchema
from intervention_graph_creation.src.local_graph_extraction.core.local_graph import LocalGraph
from intervention_graph_creation.src.local_graph_extraction.core.edge import GraphEdge
from intervention_graph_creation.src.local_graph_extraction.core.node import GraphNode
from intervention_graph_creation.src.local_graph_extraction.db.helpers import label_for

SETTINGS = load_settings()

class AISafetyGraph:
    def __init__(self) -> None:
        self.db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)

    # ---------- nodes ----------

    def upsert_node(self, node: GraphNode, url: str) -> None:
        g = self.db.select_graph(SETTINGS.falkordb.graph)
        base_label = label_for(node.type)            # "Concept" or "Intervention"
        generic_label = "NODE"

        # Prepare params
        params = {
            "name": node.name,
            "type": node.type,
            "description": node.description,
            "aliases": node.aliases,
            "concept_category": node.concept_category,
            "intervention_lifecycle": node.intervention_lifecycle,
            "intervention_maturity": node.intervention_maturity,
            "url": url,
            "is_tombstone": node.is_tombstone,
            "is_leaf": node.is_leaf,
            "cycled_id": node.cycled_id or "",
            "created_at": node.created_at,
            "updated_at": node.updated_at,
            "embedding": (node.embedding.tolist() if node.embedding is not None else None),
        }

        # - MERGE uses ONLY the base label so existing nodes (without :NODE) still match.
        # - We add :NODE after MERGE via SET n:NODE.
        # - For embedding, we conditionally set vecf32(...) only when provided; otherwise set NULL.
        # - Keep url as property AND create [:FROM] relationship to :Source node
        cypher = f"""
        MERGE (n:{base_label} {{name: $name, type: $type}})
        SET n:{generic_label},
            n.description = $description,
            n.aliases = $aliases,
            n.concept_category = $concept_category,
            n.intervention_lifecycle = $intervention_lifecycle,
            n.intervention_maturity = $intervention_maturity,
            n.url = $url,
            n.is_tombstone = $is_tombstone,
            n.is_leaf = $is_leaf,
            n.cycled_id = $cycled_id,
            n.created_at = coalesce(n.created_at, $created_at),
            n.updated_at = $updated_at
        WITH n, $embedding AS emb
        SET n.embedding = CASE WHEN emb IS NULL THEN NULL ELSE vecf32(emb) END
        WITH n,  $url AS source_url
        MERGE (p:Source {{url: source_url}})
        MERGE (n)-[:FROM]->(p)
        RETURN n
        """

        g.query(cypher, params)

    # ---------- edges ----------
    # Multiple edges between same nodes are allowed,
    # but for the same type we update the existing edge (MERGE by type).

    def upsert_edge(self, edge: GraphEdge, url: str) -> None:
        g = self.db.select_graph(SETTINGS.falkordb.graph)

        # Prepare params
        params = {
            "s": edge.source_node or "",
            "t": edge.target_node or "",
            "type": edge.type or "",
            "description": edge.description or "",
            "edge_confidence": edge.edge_confidence,
            "url": url,
            "is_tombstone": edge.is_tombstone,
            "merge_rational": edge.merge_rational or "",
            "cycled_id": edge.cycled_id or "",
            "created_at": edge.created_at,
            "updated_at": edge.updated_at,
            "embedding": (edge.embedding.tolist() if edge.embedding is not None else None)
        }

        # Assume nodes already exist with correct labels; do not create them here.
        # One :EDGE per (a,b,type). If exists → update props; else → create.
        # Keep url as property AND create [:FROM] relationship to :Source node

        # First, create/update the edge
        edge_cypher = f"""
        MATCH (a {{name: $s}}), (b {{name: $t}})
        MERGE (a)-[r:EDGE {{type: $type}}]->(b)
        SET r.description = $description,
            r.edge_confidence = $edge_confidence,
            r.url = $url,
            r.is_tombstone = $is_tombstone,
            r.merge_rational = $merge_rational,
            r.cycled_id = $cycled_id,
            r.created_at = coalesce(r.created_at, $created_at),
            r.updated_at = $updated_at
        WITH r, $embedding AS emb
        SET r.embedding = CASE WHEN emb IS NULL THEN NULL ELSE vecf32(emb) END
        RETURN ID(r) AS edge_id
        """

        result = g.query(edge_cypher, params)
        edge_id = result.result_set[0][0] if result.result_set else None

        # Then create the relationship from edge to source node
        if edge_id is not None:
            rel_cypher = f"""
            MATCH (r) WHERE ID(r) = {edge_id}
            MERGE (p:Source {{url: $url}})
            MERGE (r)-[:FROM]->(p)
            """
            g.query(rel_cypher, {"url": params["url"]})


    def ingest_metadata(self, metadata: List[Dict[str, Any]]) -> None:
        """Ingest metadata as :Source nodes in the graph."""
        g = self.db.select_graph(SETTINGS.falkordb.graph)

        # Convert metadata list to dictionary for easier processing
        meta_dict = {item.key: item.value for item in metadata}

        # Extract all metadata fields from META_KEYS
        url = meta_dict.get("url", "")
        title = meta_dict.get("title", "")
        authors = meta_dict.get("authors", [])
        date_published = meta_dict.get("date_published", "")
        source = meta_dict.get("source", "")
        filename = meta_dict.get("filename", "")
        source_filetype = meta_dict.get("source_filetype", "")

        # Ensure authors is a list
        if isinstance(authors, str):
            authors = [authors]


        cypher = """
        MERGE (p:Source {url: $url})
        SET p.title = $title,
            p.authors = $authors,
            p.date_published = $date_published,
            p.source = $source,
            p.filename = $filename,
            p.source_filetype = $source_filetype
        RETURN p
        """

        params = {
            "url": url,
            "title": title,
            "authors": authors,
            "date_published": date_published,
            "source": source,
            "filename": filename,
            "source_filetype": source_filetype
        }

        g.query(cypher, params)

    def ingest_rationale(self, rationale_path: Path, url: str) -> None:
       """Ingest rationale record as :Rationale node linked to :Source node.
       """
       # Inputs are base path ending with *_summary.txt; associated JSON is stem without suffix
       summary_path = rationale_path
       json_path = rationale_path.with_name(rationale_path.name.replace("_summary.txt", ".json"))


       if not summary_path.exists() and not json_path.exists():
           return


       g = self.db.select_graph(SETTINGS.falkordb.graph)


       # Read summary and JSON content if present
       summary_content = ""
       if summary_path.exists():
           with open(summary_path, 'r', encoding='utf-8') as f:
               summary_content = f.read().strip()


       json_content = ""
       if json_path.exists():
           try:
               with open(json_path, 'r', encoding='utf-8') as f:
                   json_dict = json.load(f)
               json_content = json.dumps(json_dict, ensure_ascii=False, indent=2)
           except Exception:
               with open(json_path, 'r', encoding='utf-8') as f:
                   json_content = f.read().strip()


       combined_parts = []
       if summary_content:
           combined_parts.append("# Summary\n\n" + summary_content)
       if json_content:
           combined_parts.append("# JSON Output\n\n" + json_content)
       combined_content = "\n\n---\n\n".join(combined_parts) if combined_parts else ""


       cypher = """
       MERGE (p:Source {url: $url})
       MERGE (r:Rationale {url: $url})
       SET r.content = $content
       MERGE (p)-[:HAS_RATIONALE]->(r)
       RETURN r
       """


       params = {
           "url": url,
           "content": combined_content,
       }


       g.query(cypher, params)

    # ---------- indexes ----------

    def set_index(self) -> None:
        g = self.db.select_graph(SETTINGS.falkordb.graph)

        # Check for existing vector index on (n:NODE).embedding
        result = g.ro_query("CALL db.indexes()")
        index_exists = False
        for row in result.result_set:
            if (
                (len(row) >= 3)
                and ("NODE" in str(row[0]))
                and ("embedding" in str(row[1]))
                and ("VECTOR" in str(row[2]).upper())
            ):
                index_exists = True
                break
        if index_exists:
            print("Dropping existing vector index on (n:NODE).embedding...")
            try:
                g.query("DROP VECTOR INDEX FOR (n:NODE) ON (n.embedding)")
            except Exception as e:
                print(f"Warning: Failed to drop vector index (may not exist or not supported): {e}")
                
        print("Creating new vector index on (n:NODE).embedding...")
        g.query("CREATE VECTOR INDEX FOR (n:NODE) ON (n.embedding) OPTIONS {dimension:1536, similarityFunction:'cosine'}")
        print("Created vector index on (n:NODE).embedding.")

        # Check for existing vector index on [r:EDGE].embedding
        result = g.ro_query("CALL db.indexes()")
        edge_index_exists = False
        for row in result.result_set:
            if (
                (len(row) >= 3)
                and ("EDGE" in str(row[0]))
                and ("embedding" in str(row[1]))
                and ("VECTOR" in str(row[2]).upper())
            ):
                edge_index_exists = True
                break
        if edge_index_exists:
            print("Dropping existing vector index on [r:EDGE].embedding...")
            try:
                g.query("DROP VECTOR INDEX FOR ()-[r:EDGE]-() ON (r.embedding)")
            except Exception as e:
                print(f"Warning: Failed to drop vector index (may not exist or not supported): {e}")
        print("Creating new vector index on [r:EDGE].embedding...")
        g.query("CREATE VECTOR INDEX FOR ()-[r:EDGE]-() ON (r.embedding) OPTIONS {dimension:1536, similarityFunction:'cosine'}")
        print("Created vector index on (r:EDGE).embedding.")

    # ---------- ingest ----------

    def clear_graph(self) -> None:
        """Delete all nodes and edges from the graph"""
        g = self.db.select_graph(SETTINGS.falkordb.graph)  # ← Use the settings

        print(f"Clearing all data from graph '{SETTINGS.falkordb.graph}'...")

        # Delete all relationships first, then nodes
        g.query("MATCH ()-[r]-() DELETE r")
        g.query("MATCH (n) DELETE n")

        print("Graph cleared successfully")

    def clear_and_ingest(self, input_dir: Path) -> None:
        """Clear existing data and ingest new data"""
        self.clear_graph()
        self.ingest_dir(input_dir)

    def ingest_file(self, json_path: Path, errors: dict) -> bool:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        # Filter out metadata entries with None values to avoid validation errors
        if 'meta' in data and data['meta']:
            data['meta'] = [meta for meta in data['meta'] if meta.get('value') is not None]

        doc = PaperSchema(**data)

        # Extract URL from metadata for use in rationale ingestion
        meta_dict = {item.key: item.value for item in doc.meta}
        url = meta_dict.get("url", "")

        # Ingest metadata before the local graph
        self.ingest_metadata(doc.meta)

        # Ingest rationale if available
        # Ingest rationale if available
        rationale_path = json_path.with_stem(json_path.stem + '_summary').with_suffix('.txt')
        if rationale_path.exists() or json_path.exists():
            self.ingest_rationale(rationale_path, url)

        local_graph, error_msg = LocalGraph.from_paper_schema(doc, json_path)
        if local_graph is None:
            # Error already logged by from_paper_schema
            errors[json_path.stem] = [error_msg] if error_msg else ["Invalid paper: see error log for details."]
            return True
        for node in local_graph.nodes:
            local_graph.add_embeddings_to_nodes(node, json_path)
        for edge in local_graph.edges:
            local_graph.add_embeddings_to_edges(edge, json_path)
        self.ingest_local_graph(local_graph, url)

        return False

    def ingest_dir(self, input_dir: Path = SETTINGS.paths.output_dir) -> None:
        errors = {}
        base = Path(input_dir)
        issues_dir = SETTINGS.paths.graph_error_dir
        issues_dir.mkdir(exist_ok=True)
        subdirs = [d for d in base.iterdir() if d.is_dir()]

        for d in tqdm(sorted(subdirs)):
            json_path = d / f"{d.name}.json"
            if not json_path.exists():
                print(f"⚠️ Skipping {d.name}: {json_path} not found")
                continue

            has_issue = self.ingest_file(json_path, errors)
            if has_issue:
                target_dir = issues_dir / d.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(d), str(issues_dir))

        # Write all errors to issues/issues.json
        if errors:
            issues_json_path = issues_dir / "issues.json"
            with open(issues_json_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
            print("\n=== Files with issues ===")
            for k, v in errors.items():
                print(f"- {k}.json: {', '.join(v)}")

        self.set_index()

    def ingest_local_graph(self, local_graph: LocalGraph, url: str) -> None:
        for node in local_graph.nodes:
            self.upsert_node(node, url)
        for edge in local_graph.edges:
            self.upsert_edge(edge, url)

    # ---------- utils ----------

    def get_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        g = self.db.select_graph(SETTINGS.falkordb.graph)

        node_res = g.ro_query("MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, n AS node")
        nodes = []
        for row in node_res.result_set:
            node_id = row[0]
            labels = row[1] or []
            node = row[2]
            props = node.properties or {}
            parts = []
            for k, v in props.items():
                if k == "id":
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
            nodes.append({"id": node_id, "labels": labels, "text": text})

        edge_res = g.ro_query(
            "MATCH (n)-[r]->(m) RETURN ID(r) AS id, TYPE(r) AS type, ID(n) AS source, ID(m) AS target, r AS rel"
        )
        edges = []
        for row in edge_res.result_set:
            edge_id = row[0]
            edge_type = row[1]
            source = row[2]
            target = row[3]
            rel = row[4]
            props = rel.properties or {}
            edges.append(
                {
                    "id": edge_id,
                    "type": edge_type,
                    "source": source,
                    "target": target,
                    "properties": props,
                }
            )

        return {"nodes": nodes, "edges": edges}

    def save_graph_to_json(self, filepath: str) -> None:
        data = self.get_graph()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def merge_nodes(self, keep_name: str, remove_name: str):
        """
        Merge two nodes identified by name.
        Moves all relationships from remove_name -> keep_name, then deletes remove_name.
        """
        graph = self.db.select_graph(SETTINGS.falkordb.graph)

        # 1) Discover all relationship types touching the node to be removed (parameterized)
        rel_types_q = """
        MATCH (n {name: $remove})
        OPTIONAL MATCH (n)-[r]->() RETURN DISTINCT type(r) AS t
        UNION
        MATCH (n {name: $remove})
        OPTIONAL MATCH ()-[r]->(n) RETURN DISTINCT type(r) AS t
        """
        res = graph.query(rel_types_q, {"remove": remove_name})
        rel_types = [row[0] for row in res.result_set if row[0]]

        # If no relationships, just delete the node (parameterized)
        if not rel_types:
            return graph.query("MATCH (a {name: $remove}) DELETE a", {"remove": remove_name})

        # 2) Build the merge query dynamically with the discovered relationship types.
        #    NOTE: relationship *types* cannot be parameterized in Cypher.
        parts = []
        for rtype in rel_types:
            parts.append(f"""
            // Move outgoing :{rtype}
            OPTIONAL MATCH (a {{name: $remove}})-[r:{rtype}]->(m)
            MATCH (b {{name: $keep}})
            FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
                MERGE (b)-[r2:{rtype}]->(m)
                SET r2 += r
            )
            WITH a, b

            // Move incoming :{rtype}
            OPTIONAL MATCH (m2)-[s:{rtype}]->(a {{name: $remove}})
            FOREACH (_ IN CASE WHEN m2 IS NULL THEN [] ELSE [1] END |
                MERGE (m2)-[s2:{rtype}]->(b)
                SET s2 += s
            )
            WITH a, b
            """)

        merge_q = f"""
        MATCH (a {{name: $remove}}), (b {{name: $keep}})
        {"".join(parts)}
        DELETE a
        """

        return graph.query(merge_q, {"remove": remove_name, "keep": keep_name})


def main():
    print("starting script")
    graph = AISafetyGraph()
    print(
        f"Starting ingestion into FalkorDB graph '{SETTINGS.falkordb.graph}' at {SETTINGS.falkordb.host}:{SETTINGS.falkordb.port}..."
    )
    graph.clear_graph()
    print(f"Ingesting from {SETTINGS.paths.output_dir}...")
    graph.ingest_dir(SETTINGS.paths.output_dir)
    print("Ingestion complete, saving graph to json...")
    graph.save_graph_to_json("graph.json")


if __name__ == "__main__":
    main()
