import os
import json
from pathlib import Path

from .extractor import Extractor

INPUT_JSON_FILE = Path('intervention_graph_creation/data/raw/ard_json_test/agentmodels.jsonl')
INPUT_JSON_DIR = Path('intervention_graph_creation/data/raw/ard_json_test')
INPUT_PDF_DIR = Path('intervention_graph_creation/data/raw/pdfs_local')

extractor = Extractor()

# extractor.process_jsonl(INPUT_JSON_FILE)
extractor.process_dir(INPUT_JSON_DIR)
# extractor.process_dir(INPUT_PDF_DIR, 2)