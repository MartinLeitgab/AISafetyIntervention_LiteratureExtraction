### LLM-Assisted Graph Merging — Step 1: Prepare Inputs for the LLM

This document explains how the Step 1 code prepares compact, data-only inputs
for LLM comparisons between two nodes (and lays the groundwork for edges later).
It focuses solely on preparing the data; there are no prompts, LLM calls, or merge actions here.


## Goals (Step 1)
- Provide the LLM enough context to decide if two nodes should be merged.
- Data included per comparison:
  - Node text (surface form)
  - Aliases / alternate names
  - Context / reasoning (why the nod# Ensure project root is on sys.path so `src` is importable when running from repo root
e exists): node notes + rationales from linked edges
  - Source metadata (paper_id, with optional fields reserved for title/section/paragraph)
  - Linked edges (immediate connections with type, rationale, confidence, and source)


## Files Overview
- `<...>/src/merge_types.py`
  - `SourceMetadata`: identifies where a node/edge came from (currently `paper_id` from the output filename; placeholders for `title`, `section`, `paragraph_id`).
  - `LinkedEdgeSummary`: short record of an immediate connection (edge type, rationale, confidence, and source).
  - `NodeAggregate`: collected per-node data from all occurrences across outputs.
  - `NodeViewForComparison`: the per-node view handed to the LLM later (text, aliases, context, sources, linked edges).
  - `NodeComparisonInput`: a pair of `NodeViewForComparison` (data-only) for A vs B.
  - Edge types (`EdgeViewForComparison`, `EdgeComparisonInput`) are defined for a later step.

- `<...>/llm_assisted_graph_merging/src/merge_indexer.py`
  - `build_merge_index(output_dir)`: reads parsed `OutputSchema` JSON files from `output/`, aggregates nodes by canonical name (lowercased), merges aliases and notes, and collects linked edge rationales as context.
  - Produces `MergeIndex` with `nodes: Dict[node_key, NodeAggregate]`.

- `<...>/llm_assisted_graph_merging/src/merge_input_builder.py`
  - `build_node_comparison_input(index, key_a, key_b)`: converts two aggregates to a data-only `NodeComparisonInput`.
  - The node view includes:
    - `text`, `aliases`
    - `context` (node notes + "[EDGE_TYPE] rationale" for immediate connections)
    - `source_metadata`
    - `linked_edges`

- `<...>/llm_assisted_graph_merging/examples/walkthrough_prepare_llm_input.py`
  - Runnable script demonstrating the end-to-end Step 1 flow.


## How Requirements Are Met
- Node text: `NodeViewForComparison.text`
- Aliases: `NodeViewForComparison.aliases`
- Context / reasoning: `NodeViewForComparison.context` combines node notes with connected edge rationales.
- Source metadata: `NodeViewForComparison.source_metadata` is a list of `SourceMetadata` (currently `paper_id`; placeholders for title/section/paragraph).
- Linked edges: `NodeViewForComparison.linked_edges` contains `LinkedEdgeSummary` for each immediate connection.

There is deliberately no prompt or instruction text in Step 1. This is data-only.


## Walkthrough
- List available node keys and build a comparison payload:
```bash
uv run python ardhito/llm_assisted_graph_merging/examples/walkthrough_prepare_llm_input.py --list --limit 10
uv run python ardhito/llm_assisted_graph_merging/examples/walkthrough_prepare_llm_input.py
uv run python ardhito/llm_assisted_graph_merging/examples/walkthrough_prepare_llm_input.py --key-a <node_key_a> --key-b <node_key_b>
uv run python ardhito/llm_assisted_graph_merging/examples/walkthrough_prepare_llm_input.py --save output/node_comparison_example.json
```

Note: The example script prepends its package directory to `sys.path` so imports are username-free (e.g., it imports `src.*` rather than `ardhito.*`). This makes the walkthrough runnable from the repository root without introducing usernames into import paths.

The printed (or saved) JSON is the exact payload you can provide to an LLM in Step 2.


## How This Enables Step 2 (Preview Only)
- Candidate selection can embed `NodeAggregate` fields (canonical name, text, aliases, top-k notes) and return top-N similar `node_key` pairs.
- LLM evaluation consumes `NodeComparisonInput` and returns MERGE vs KEEP SEPARATE with a reason (Step 2.2, not implemented here).
- Merge execution uses stable `node_key`s to unify aliases/notes and repoint edges.