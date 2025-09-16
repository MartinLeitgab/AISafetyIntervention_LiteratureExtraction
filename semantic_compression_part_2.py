# pyright: standard
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
        (length(p) = 0 AND NOT (n)-[:EDGE]-())
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

    # Create a self-loop edge on Node 1 using MERGE
    
  
    #Pass the information and an appropriate compression prompt (to be developed) to the compression LLM for combining of the set (creates one or more parent nodes from nodes in the set and returns JSON, with data structure as laid out in @MartinLeitgab 's comment to @ArdhitoN 's PR LLM-assisted Graph-Merging - Step 1 - data-only LLM input prep for node comparisons eamag/AISafetyIntervention_LiteratureExtraction#1 (comment)). (maybe partially implemented
    # ii.
    context = get_prompt_for_merge_judge(result.result_set)
    merge_sets = merge_judge(context)


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


# TODO Things to test in unit tests:
# part i.
# - Test that all nodes in the cluster are included
# - Test that all edges between the nodes in the cluster are included
# - Test that multihop doesn't happen, only direct connections are found
# - Test that tombstone nodes and edges are not included
# - Test that isolated nodes (no edges) are included
# - Test that if both nodes in an edge are in the cluster, the edge is included only once
# - What if a node in the cluster has an edge to itself (self-loop)?
#   create self_loop_query = """
#     MATCH (n:NODE)
#     WHERE id(n) = 1
#     MERGE (n)-[r:EDGE {is_tombstone: false}]->(n)
#     """
if __name__ == "__main__":

    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    
    g = db.select_graph(SETTINGS.falkordb.graph)


    # Example cluster of similar node IDs
    example_cluster = [1, 2, 3, 133]
    
    # Compress the cluster
    compress_cluster(g, example_cluster)