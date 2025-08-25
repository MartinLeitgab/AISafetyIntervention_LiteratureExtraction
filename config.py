from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "./intervention_graph_creation/data/raw/pdfs_local"
OUTPUT_DIR = PROJECT_ROOT / "./intervention_graph_creation/data/processed"

FALKORDB_PORT = 6379
FALKORDB_HOST = "localhost"
FALKORDB_GRAPH = "AISafetyIntervention"

GRAPH_HTML_OUT = "graph.html"
