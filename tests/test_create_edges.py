# pyright: standard

from config import load_settings
from dataclasses import dataclass
from falkordb import FalkorDB, Graph
from intervention_graph_creation.src.local_graph_extraction.compression.create_edges import create_edges

SETTINGS = load_settings()

@dataclass
class GraphFixture:
    graph: Graph

def copy_graph(new_graph_name: str) -> Graph:
    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    graph = db.select_graph(SETTINGS.falkordb.graph)
    return graph.copy(new_graph_name)



def test_1():
    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    graph = db.select_graph(SETTINGS.falkordb.graph)
    create_edges(graph, 'SIMILARITY_ABOVE_POINT_EIGHT', min_cosine_similarity=0.8)
   