import json
from pathlib import Path
from typing import List

"""How well did the judge perform. Did it extract what we expected?"""
from fire import Fire  # type: ignore[import]
from judge import JudgeReport
# pyright: basic
import numpy as np
from schema import AddNodeFix


def grab_all_json(output_path: Path):
    p = Path(output_path)
    if not p.exists():
        raise FileNotFoundError(f"Output path does not exist: {output_path}")

    json_files = sorted(str(pf) for pf in p.rglob("*.json"))
    out_files: List[str] = []
    for jf in json_files:
        # print(jf)
        if "errors.json" in jf or "summary.json" in jf:
            continue
        out_files.append(jf)
    return out_files
def main(output_path: str):
    json_files = grab_all_json(Path(output_path))

    add_nodes_counts = []
    add_edges_counts = []

    for jf in json_files:
        with open(jf, 'r') as f:
            data : JudgeReport = json.load(f)
        add_nodes = data["proposed_fixes"]["add_nodes"]
        if add_nodes is None:
            continue
        # add_nodes_counts.append(len(add_nodes))
        add_node_count = 0
        edge_count = 0
        for add_node_raw in add_nodes:
            if add_node_raw is None:
                continue
            try:
                add_node_fix = AddNodeFix.model_validate(add_node_raw["data"])
                add_node_count += 1
                for edge in add_node_fix.new_edges:
                    if edge.data is None:
                        continue
                    if edge.data.get_other_node_name(add_node_fix.new_node.name) is None:
                        continue
                    edge_count += 1
            except Exception:
                pass
        add_nodes_counts.append(add_node_count)
        add_edges_counts.append(edge_count)
    # np.mean(add_nodes_counts)
    # np.mean(add_edges_counts)
    print(f"Average number of added nodes: {np.mean(add_nodes_counts)} std {np.std(add_nodes_counts)}, median {np.median(add_nodes_counts)}. \n Max {np.max(add_nodes_counts)}, min {np.min(add_nodes_counts)}")
    print(f"Average number of added edges: {np.mean(add_edges_counts)} std {np.std(add_edges_counts)}, median {np.median(add_edges_counts)}. \n Max {np.max(add_edges_counts)}, min {np.min(add_edges_counts)}")

    print("If there is an added node then:")
    add_node_at_least_one = [x for x in add_nodes_counts if x > 0]
    print(f"  Average number of added nodes: {np.mean(add_node_at_least_one)} std {np.std(add_node_at_least_one)}, median {np.median(add_node_at_least_one)}")
       
    

if __name__ == "__main__":
    Fire(main)