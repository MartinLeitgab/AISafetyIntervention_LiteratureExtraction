import sys
from falkordb import FalkorDB
import networkx as nx
from pyvis.network import Network

from config import FALKORDB_GRAPH, FALKORDB_HOST, FALKORDB_PORT, GRAPH_HTML_OUT


def fetch_all(client, graph_name):
    g = client.select_graph(graph_name)
    nres = g.ro_query("MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, n AS n")
    eres = g.ro_query("MATCH (a)-[r:EDGE]->(b) RETURN ID(a) AS sid, ID(b) AS tid, r AS r")

    nodes = []
    for row in nres.result_set:
        nid, labels, n = row
        props = n.properties or {}
        nodes.append({
            "id": nid,
            "name": props.get("name") or f"node_{nid}",
            "labels": labels or [],
            "type": (props.get("type") or "").lower(),  # "concept" / "intervention"
            "props": props
        })

    edges = []
    for row in eres.result_set:
        sid, tid, r = row
        rprops = getattr(r, "properties", {}) or {}
        edges.append({
            "source": sid,
            "target": tid,
            "etype": rprops.get("etype"),
            "description": rprops.get("description"),
            "edge_confidence": rprops.get("edge_confidence"),
            "props": rprops
        })
    return nodes, edges


def build_nx(nodes, edges):
    G = nx.DiGraph()

    for n in nodes:
        G.add_node(
            n["id"],
            label=n["name"],
            ntype=n["type"],
            **(n["props"] or {}),
        )

    for e in edges:
        props = dict(e.get("props") or {})
        # remove duplicates that we set explicitly below
        for k in ("etype", "description", "edge_confidence"):
            props.pop(k, None)

        G.add_edge(
            e["source"], e["target"],
            etype=e.get("etype"),
            description=e.get("description"),
            edge_confidence=e.get("edge_confidence"),
            **props,
        )
    return G


def render_pyvis(G, outfile="graph.html"):
    net = Network(height="800px", width="100%", bgcolor="#ffffff", directed=True, notebook=False)
    net.toggle_physics(True)

    color_by_type = {"concept": "#2563eb", "intervention": "#16a34a"}
    for nid, data in G.nodes(data=True):
        title_lines = [f"<b>{data.get('label','(no name)')}</b>"]
        for k in ("type","concept_category","intervention_lifecycle","intervention_maturity","paper_id","description"):
            if k in data and data[k] not in (None, ""):
                title_lines.append(f"{k}: {data[k]}")
        if "aliases" in data and data["aliases"]:
            try:
                alias_str = ", ".join(map(str, data["aliases"]))
            except Exception:
                alias_str = str(data["aliases"])
            title_lines.append("aliases: " + alias_str)
        title = "<br>".join(title_lines)

        net.add_node(
            nid,
            label=data.get("label",""),
            title=title,
            color=color_by_type.get(data.get("ntype"), "#6b7280"),
            shape="dot",
            size=12,
        )

    for i, (u, v, data) in enumerate(G.edges(data=True)):
        et = data.get("etype") or "EDGE"
        conf = data.get("edge_confidence")
        desc = data.get("description")
        tooltip_parts = [f"<b>{et}</b>"]
        if conf is not None:
            tooltip_parts.append(f"confidence: {conf}")
        if desc:
            tooltip_parts.append(desc)
        title = "<br>".join(tooltip_parts)

        net.add_edge(
            u, v,
            label=str(et),
            title=title,
            arrows="to",
            smooth={"type": "curvedCW", "roundness": (0.1 + (i % 5) * 0.05)}
        )

    # Use write_html instead of show to avoid template.render issue
    net.write_html(outfile, notebook=False)
    return outfile


def main():
    print(f"[viz] Connecting to {FALKORDB_HOST}:{FALKORDB_PORT}, graph={FALKORDB_GRAPH}")
    client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    nodes, edges = fetch_all(client, FALKORDB_GRAPH)
    print(f"[viz] Fetched {len(nodes)} nodes, {len(edges)} edges")
    G = build_nx(nodes, edges)
    out = render_pyvis(G, GRAPH_HTML_OUT)
    print(f"[viz] Wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[viz] ERROR:", e, file=sys.stderr)
        sys.exit(1)
