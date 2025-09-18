# pyright: standard
#TODO rename file and move to appropriate location
from falkordb import FalkorDB, Graph, Path, Node, Edge

from typing import List, Set, Optional
from config import load_settings
import os
import json
from typing import List, Set
from dataclasses import dataclass, field
import re
from pathlib import Path as PathLibPath
from openai import OpenAI
from os import environ

import dotenv
dotenv.load_dotenv()

SETTINGS = load_settings()

@dataclass
class MergeSet:
    """
    A set of node IDs to be merged, along with a rationale for the merge decision.
    Each MergeSet should contain at least two node IDs.
    Each node ID should be unique across all MergeSets, ensuring no node ID appears in more than one set.
    """
    nodes: Set[int]
    rationale: str
    parameters: dict = field(default_factory=dict)
    def to_dict(self):
        """Convert the MergeSet to a dictionary for JSON serialization."""
        return {
            'nodes': list(self.nodes),
            'rationale': self.rationale,
            'parameters': self.parameters
        }



def get_prompt_for_merge_llm(cluster_paths: List[Path], primary_node_ids: List[int]) -> str:
    """
    Given a list of paths representing the context of a cluster of similar nodes,
    generate a textual context to be provided to the merge judge LLM.
    Each path contains nodes and edges, which can be used to extract relevant information.
    """
    node_infos = []
    edge_infos = []
    node_ids = set()

    nodes: dict[int, Node] = {}
    for path in cluster_paths:
        for node in path.nodes():
            node_ids.add(node.id)
            nodes[node.id if node.id is not None else -1] = node
        for edge in path.edges():
            edge_infos.append(f"Edge: { edge.properties.get('type', 'unknown')} from {edge.src_node} to {edge.dest_node}")

    for node_id, node in sorted(nodes.items()):
        node_infos.append(
            f"Node ID: {node_id}\n"
            f"Name: {node.properties.get('name', 'unknown')}\n"
            f"Type: {node.properties.get('type', 'unknown')}\n"
            f"Description: {node.properties.get('description', 'unknown')}\n"
        )


    prompt = (
        "# AI Safety Knowledge Graph Semantic Compression\n"
        "You are an expert in AI safety knowledge graph compression. Given the following nodes and their relationships, your task is to:\n"
        "1. Only consider merging the primary nodes listed below. Do NOT merge or suggest merging any neighbor nodes.\n"
        "2. Decide which primary nodes should be merged into a single supernode (merged concept).\n"
        "3. Provide a clear rationale for each merge decision. Reason step by step\n"
        "4. For each merge set, generate merged parameters for the supernode: name, description, type, and any other relevant attributes.\n\n"
        f"Consider only these nodes for merging: {primary_node_ids}\n\n"
        "Nodes:\n" + "\n".join(node_infos) +
        "\n\nEdges:\n" + "\n".join(edge_infos) +
        "\n\nOutput Instructions:\n"
        "Return ONLY a valid JSON object with the following format:\n"
        "{\n"
        "  \"merge_sets\": [\n"
        "    {\n"
        "      \"node_ids\": [list of node IDs to merge],\n"
        "      \"rationale\": \"reason for merging\",\n"
        "      \"parameters\": {\n"
        "        \"name\": \"string - concise name for the supernode\",\n"
        "        \"type\": \"string - node type\",\n"
        "        \"description\": \"string - comprehensive description\",\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return prompt


def merge_llm(context: str) -> List[MergeSet]:
    """
    Calls an LLM to decide which nodes to merge and why.
    Returns a list of MergeSet objects.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found. Returning empty merge sets.")
        return []

    client = OpenAI(api_key=api_key)
    # model = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    model = os.environ.get("OPENAI_MODEL_NAME", "o3")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert AI safety knowledge graph compression assistant."},
            {"role": "user", "content": context}
        ],
        # temperature=0.1,
        # max_tokens=2000
        # max_tokens=70000 not compaitible with o series models
        # max_tokens=4096 # gpt-3.5-turbo limit is 4096
        # max_tokens=8000 # o1/o3 limit is 8192
    )
    if environ.get("SAVE_DEBUG_DATA") == "1":
        out_path = PathLibPath("./test_output_data/merge_llm_response.json")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response.model_dump_json())
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM response has no content.")
    # First, try to parse the whole response as JSON
    try:
        result = json.loads(content)
        if "merge_sets" in result:
            merge_sets = []
            for ms in result.get("merge_sets", []):
                merge_sets.append(MergeSet(nodes=set(ms["node_ids"]), rationale=ms["rationale"], parameters=ms.get("parameters", {})))
            return merge_sets
    except Exception:
        pass
    # Try to extract code-fenced JSON (```json ... ```
    code_json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if code_json_match:
        json_str = code_json_match.group(1)
        try:
            result = json.loads(json_str)
            if "merge_sets" in result:
                merge_sets = []
                for ms in result.get("merge_sets", []):
                    merge_sets.append(MergeSet(nodes=set(ms["node_ids"]), rationale=ms["rationale"], parameters=ms.get("parameters", {})))
                return merge_sets
        except Exception:
            pass
    # Fallback: extract first JSON object using regex
    json_candidates = re.findall(r'\{[\s\S]*?\}', content)
    for json_str in json_candidates:
        try:
            result = json.loads(json_str)
            if "merge_sets" in result:
                merge_sets = []
                for ms in result.get("merge_sets", []):
                    merge_sets.append(MergeSet(nodes=set(ms["node_ids"]), rationale=ms["rationale"], parameters=ms.get("parameters", {})))
                return merge_sets
        except Exception:
            continue
    print("Error: No valid merge_sets JSON found in LLM response.")
    print("Raw response:", content)
    raise ValueError("No valid merge_sets JSON found in LLM response.")
    # return []


def get_cluster_paths(g: Graph, cluster_nodes: List[int]) -> List[List[Path]]:
    """
    Given a graph and a list of node IDs representing a cluster,
    retrieve all paths that include these nodes.

    Each path should include nodes and edges, which can be used to extract relevant information.
    """

    cypher_query = """
    // Find all paths between nodes (non-tombstone) in the cluster with length 0 or 1
    // the search is bidirectional
    MATCH p=(n:NODE {is_tombstone:false})-[:EDGE*0..1 {is_tombstone:false}]-(m:NODE {is_tombstone:false})
    // id of n must be in the cluster
    WHERE id(n) IN $cluster
      AND (
        // if path length is 0, n is isolated, include it
        (length(p) = 0 AND NOT (n)-[:EDGE {is_tombstone:false}]-() OR (n)-[:EDGE]-(:NODE {is_tombstone: true}))
        OR
        (length(p) = 1 AND (
                NOT id(m) IN $cluster 
                OR
                // only include the edge once (id(n) <= id(m)) to avoid duplicates.
                // Use <= instead of < to allow self-loops
                id(n) <= id(m)
            )
        )
      )
    // Need to return *distinct* to avoid duplicates when
    // we have self-loops
    RETURN DISTINCT p
    """
    result = g.query(cypher_query, {"cluster": cluster_nodes})
    return result.result_set

# https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/100
# For each set of very similar nodes execute the following steps
def compress_cluster(g: Graph, cluster_nodes: List[int]):
    # Use vlr to find only the directly connected nodes
    # https://docs.falkordb.com/cypher/match#variable-length-relationships
    # 0 is the minimum length (include the node itself if it has no edges)
    # 1 is the maximum length (only directly connected nodes)

    # and use bidirectional path traversal
    # to find all connected nodes regardless of which is source or target
    # https://docs.falkordb.com/cypher/match.html#bidirectional-path-traversal

    # i. For each node in the set, collect all edge information and all 
    # immediate neighbor nodes information (only nodes of neighbors, 
    # not all edges of neighbors except the one connecting back to the node in the set here) 
    # @ArdhitoN 's code, @jonpsy ’s 
    # suggestion in Issue [Story 4/5] 
    # Integrating database-driven approach for supernodes and superedges #35

    result = get_cluster_paths(g, cluster_nodes)
    # now results should return 1 path for each edge between nodes in the cluster
    # and 1 path for each isolated node in the cluster

    #Pass the information and an appropriate compression prompt to the LLM
    context = get_prompt_for_merge_llm(result, cluster_nodes)
    merge_sets = merge_llm(context)


    # iii.
    # Now, for each MergeSet, perform the merge operation in the graph
    # Write post-merging parent node back into DB, 
    # e.g. setting tombstone flags on raw nodes that have been merged 
    # so that only the merged node is used in future queries, 
    # and keeping info of merged raw nodes store parent nodes with lists
    #  of edges/pointers to their child nodes 
    # (TBD code, 
    # maybe not implemented yet- see @jonpsy ’s suggestion in 
    # Issue [Story 4/5] Integrating database-driven approach 
    # for supernodes and superedges #35 )
    for merge_set in merge_sets:
        pass
        # Extract node and edge information
        # nodes = merge_set.nodes

        #  cypher_query = """
        #     MATCH (n:NODE {is_tombstone:false})
        #     WHERE id(n) IN $nodes
        #     MERGE (n {is_tombstone: true})-[r:MERGE {rationale: $rationale}]->(m:NODE {is_tombstone:false, is_leaf: false})
        #     """
        

    # TODO Write post-merging parent node back into DB, e.g. setting tombstone flags on raw nodes that have been merged so that only the merged node is used in future queries, and keeping info of merged raw nodes store parent nodes with lists of edges/pointers to their child nodes (TBD code, maybe not implemented yet- see @jonpsy ’s suggestion in Issue [Story 4/5] Integrating database-driven approach for supernodes and superedges #35 )



if __name__ == "__main__":

    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    g = db.select_graph(SETTINGS.falkordb.graph)

    # Example cluster of similar node IDs
    example_cluster = [1, 2, 3, 133]
    
    # Compress the cluster
    compress_cluster(g, example_cluster)

