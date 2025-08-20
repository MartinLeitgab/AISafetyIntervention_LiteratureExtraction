"""Minimal example: extract ConceptGraph from markdown papers using structured batch."""

import asyncio
import os
from typing import List, Optional, cast
from src.model_provider.openai_provider import OpenAIModelProvider
from src.prompts.prompt import CONCEPT_GRAPH_PROMPT
from dotenv import load_dotenv

from src.models.concept_graph import ConceptGraph
from src.utils import _graph_to_networkx, _save_graph_png


def _build_prompt(paper_text: str) -> str:
    return f"{CONCEPT_GRAPH_PROMPT}\n\nPaper:\n{paper_text}"


async def extract_graphs_from_md(
    md_dir: str, max_files: int
) -> List[Optional[ConceptGraph]]:
    from src.db.concept_graph_store import ConceptGraphStore

    provider = OpenAIModelProvider(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name="o3",
    )

    # Initialize FalkorDB store
    store = ConceptGraphStore(
        host=os.getenv("FALKORDB_HOST", "localhost"),
        port=int(os.getenv("FALKORDB_PORT", "6379")),
        password=os.getenv("FALKORDB_PASSWORD"),
    )

    filenames = sorted([f for f in os.listdir(md_dir) if f.endswith(".md")])[:max_files]
    prompts: list[str] = []
    for name in filenames:
        path = os.path.join(md_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        prompts.append(_build_prompt(text))

    results = await provider.batch_infer_structured(prompts, ConceptGraph)
    results = cast(List[Optional[ConceptGraph]], results)

    # Output directory: sibling of md_dir named "graphs"
    out_dir = os.path.join(os.path.dirname(md_dir.rstrip("/")), "graphs")
    os.makedirs(out_dir, exist_ok=True)

    for name, graph in zip(filenames, results):
        base = os.path.splitext(name)[0]
        out_png = os.path.join(out_dir, f"{base}.png")

        if graph is None:
            print(f"{name}: failed to parse ConceptGraph")
        else:
            print(f"{name}: nodes={len(graph.nodes)}, edges={len(graph.edges)}")
            # Store graph in FalkorDB
            try:
                store.store_graph(graph, paper_id=base)
                print(f"Stored graph in FalkorDB: {base}")
            except Exception as e:
                print(f"Failed to store graph in FalkorDB: {e}")

            G = _graph_to_networkx(graph)
            _save_graph_png(G, out_png)
            print(f"Saved graph PNG: {out_png}")
    return results


async def main():
    load_dotenv()
    md_dir = "/home/legacy/Research/SOAR-5/development_paper_set/md"
    await extract_graphs_from_md(md_dir, max_files=2)


if __name__ == "__main__":
    asyncio.run(main())
