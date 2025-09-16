#TODO rename file and move to appropriate location
from falkordb import FalkorDB, Graph, Path

from typing import List, Set
from config import load_settings

SETTINGS = load_settings()


class MergeSet:
    """
    A set of node IDs to be merged, along with a rationale for the merge decision.
    Each MergeSet should contain at least two node IDs.
    Each node ID should be unique across all MergeSets, ensuring no node ID appears in more than one set.
    """
    nodes: Set[int]
    rationale: str

def merge_judge(context: str) -> List[MergeSet]:
    """ 
    ii.
    A placeholder function for the merge judge logic. Ie a judge who decides which nodes to merge
    and provides a merge rationale.
    # TODO get in contact with Mitali and integrate her merge judge code from her pull request
    """
    # TODO implement merge judge from Mitali's PR
    return []

def get_prompt_for_merge_judge(cluster_paths: List[List[Path]]) -> str:
    """
    ii.
    Given a list of paths representing the context of a cluster of similar nodes,
    generate a textual context to be provided to the merge judge.

    Each path contains nodes and edges, which can be used to extract relevant information.
    """
    # TODO implement context extraction logic from Mitali's PR
    return ""

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

    cypher_query = """
    // Find all paths between nodes (non-tombstone) in the cluster with length 0 or 1
    // the search is bidirectional
    MATCH p=(n:NODE {is_tombstone:false})-[:EDGE*0..1 {is_tombstone:false}]-(m:NODE {is_tombstone:false})
    // id of n must be in the cluster
    WHERE id(n) IN $cluster
      AND (
        // if path length is 0, n is isolated, include it
        (length(p) = 0 AND NOT (n)-[:EDGE {is_tombstone:false}]-())
        OR
        // if path length is 1, include only if m is not in cluster or to avoid duplicates
        // this way a path between two nodes in the cluster is only included once
        (length(p) = 1 AND (NOT id(m) IN $cluster OR id(n) < id(m)))
      )
    RETURN p
    """
    result = g.query(cypher_query, {"cluster": cluster_nodes})
    # now results should return 1 path for each edge between nodes in the cluster
    # and 1 path for each isolated node in the cluster


    #Pass the information and an appropriate compression prompt (to be developed) to the compression LLM for combining of the set (creates one or more parent nodes from nodes in the set and returns JSON, with data structure as laid out in @MartinLeitgab 's comment to @ArdhitoN 's PR LLM-assisted Graph-Merging - Step 1 - data-only LLM input prep for node comparisons eamag/AISafetyIntervention_LiteratureExtraction#1 (comment)). (maybe partially implemented
    # ii.
    context = get_prompt_for_merge_judge(result.result_set)
    merge_sets = merge_judge(context)


    # iii.
    # Now, for each MergeSet, perform the merge operation in the graph
    for merge_set in merge_sets:
        # Extract node and edge information
        nodes = merge_set.nodes
        rationale = merge_set.rationale
        # TODO

    # TODO Write post-merging parent node back into DB, e.g. setting tombstone flags on raw nodes that have been merged so that only the merged node is used in future queries, and keeping info of merged raw nodes store parent nodes with lists of edges/pointers to their child nodes (TBD code, maybe not implemented yet- see @jonpsy ’s suggestion in Issue [Story 4/5] Integrating database-driven approach for supernodes and superedges #35 )


# TODO Things to test in unit tests:
# - Test that all nodes in the cluster are included
# - Test that all edges between the nodes in the cluster are included
# - Test that multihop doesn't happen, only direct connections are found
# - Test that tombstone nodes and edges are not included
# - Test that isolated nodes (no edges) are included
# - Test that if both nodes in an edge are in the cluster, the edge is included only once
if __name__ == "__main__":

    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    
    g = db.select_graph(SETTINGS.falkordb.graph)


    # Example cluster of similar node IDs
    example_cluster = [1, 2, 3, 133]
    
    # Compress the cluster
    compress_cluster(g, example_cluster)