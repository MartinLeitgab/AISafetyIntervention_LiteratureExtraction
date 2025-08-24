import json, os, argparse, uuid, redis
from falkordb import FalkorDB

# docker run -p 6379:6379 -p 3000:3000 -it --rm -v ./data:/var/lib/falkordb/data falkordb/falkordb:edge

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 6379))
DB_NAME = os.getenv("DB_NAME", "test")

db = FalkorDB(host=DB_HOST, port=DB_PORT)
g = db.select_graph("test")

def sanitize_props(d):
    """
    Return a dict without None values and without 'type' key.
    Values are converted to lowercase if they are strings.
    """
    out = {}
    for k, v in d.items():
        if k == "type" or v is None:
            continue
        if isinstance(v, str):
            out[k] = v.lower()
        else:
            out[k] = v
    return out

def resolve_ref(ref_value):
    """
    Resolve a source/target reference (string) to a single uid.
    Priority:
        1) exact match on JSON 'id'
        2) unique match on 'name' (only if exactly one node has that name)
    Returns uid or None.
    """
    if ref_value is None:
        return None
    # Prefer ID
    if ref_value in id_to_uid:
        return id_to_uid[ref_value]
    # Fall back to name, but only if unique
    uids = name_to_uids.get(ref_value, [])
    if len(uids) == 1:
        return uids[0]
    # Ambiguous or not found
    return None

# ---------- PASS 1: create nodes, assign UUIDs, capture mappings ----------
id_to_uid = {}       # maps JSON 'id' -> uid
name_to_uids = {}    # maps 'name' -> [uid1, uid2, ...]

for item in data['nodes']:
    node_type = item.get("type")
    if node_type not in {"concept", "intervention"}:
        continue

    label = node_type  # keep lower-case labels exactly as requested
    props = sanitize_props(item)
    uid = str(uuid.uuid4())
    props["uid"] = uid

    # Record mappings (if present)
    if "id" in item and item["id"] is not None:
        id_to_uid[item["id"]] = uid
    if "name" in item and item["name"] is not None:
        name_to_uids.setdefault(item["name"], []).append(uid)

    # CREATE (not MERGE) so duplicates (e.g., same name) are preserved
    q = f"""
    CREATE (n:`{label}`)
    SET n = $props
    RETURN n
    """
    g.query(q, {"props": props})


# ---------- PASS 2: create edges, using UUIDs ----------


skipped_edges = 0
made_edges = 0

for chain in data.get("logical_chains", []):
    for e in chain.get("edges", []):
        rel_type = e.get("type")
        src_ref = e.get("source_node")
        dst_ref = e.get("target_node")
        desc = e.get("description")
        conf = e.get("edge_confidence")

        if not (rel_type and (src_ref is not None) and (dst_ref is not None)):
            skipped_edges += 1
            continue

        src_uid = resolve_ref(src_ref)
        dst_uid = resolve_ref(dst_ref)

        if not (src_uid and dst_uid):
            # Could be ambiguous name or unknown reference — skip safely
            skipped_edges += 1
            continue

        q = f"""
        MATCH (s {{uid: $src_uid}}), (t {{uid: $dst_uid}})
        CREATE (s)-[r:`{rel_type}`]->(t)
        SET r.description = $desc, r.edge_confidence = $conf
        RETURN s, r, t
        """
        g.query(q, {"src_uid": src_uid, "dst_uid": dst_uid, "desc": desc, "conf": conf})
        made_edges += 1

# ---------- Indexes ----------
# run once upon db creation only

try:
    g.query("CREATE INDEX ON :concept(concept_category)")
except Exception as e:
    print(f"(Note) Index creation skipped or already exists: {e}")


# ---------- Summary ----------
node_count = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
edge_count = g.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
print(f"Done. Nodes in graph: {node_count}, Edges in graph: {edge_count}")
print(f"Edges created this run: {made_edges}, skipped (unresolved/ambiguous): {skipped_edges}")
print("Each node has 'uid' (UUID). Aliases preserved as a list on 'aliases'.")
print("Index attempted: CREATE INDEX ON :concept(concept_category)")