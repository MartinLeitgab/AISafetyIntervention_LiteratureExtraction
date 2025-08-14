from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from pydantic import BaseModel, ValidationError
from .prompts import OutputSchema

from .merge_types import NodeAggregate, LinkedEdgeSummary, SourceMetadata


class MergeIndex(BaseModel):
    """Aggregated, key-addressable view over nodes collected from outputs."""

    nodes: Dict[str, NodeAggregate]


def _node_key(canonical_name: str) -> str:
    return canonical_name.strip().lower()


def _iter_valid_output_files(output_dir: Path) -> Iterable[Path]:
    return sorted(p for p in output_dir.glob("*.json") if "raw_response" not in p.name)


def _load_output(path: Path) -> OutputSchema | None:
    try:
        return OutputSchema.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError):
        return None


def _summarize_edge(edge, source: SourceMetadata) -> LinkedEdgeSummary:
    return LinkedEdgeSummary(
        edge_type=edge.type,
        rationale=edge.rationale,
        confidence=edge.confidence,
        source=source,
    )


def _upsert_node_aggregate(
    aggregates: Dict[str, NodeAggregate], node, source: SourceMetadata
) -> NodeAggregate:
    key = _node_key(node.canonical_name or node.name)
    if key not in aggregates:
        aggregates[key] = NodeAggregate(
            node_key=key,
            text=node.name,
            canonical_text=node.canonical_name or node.name,
            aliases=list(node.aliases or []),
            notes=[node.notes] if getattr(node, "notes", None) else [],
            confidence_samples=[node.confidence],
            linked_edges=[],
            sources=[source],
        )
        return aggregates[key]

    agg = aggregates[key]
    if (
        node.canonical_name
        and node.canonical_name.lower() == agg.canonical_text.lower()
    ):
        agg.text = node.name
    for alias in node.aliases or []:
        if alias not in agg.aliases:
            agg.aliases.append(alias)
    note_text = getattr(node, "notes", None)
    if note_text and note_text not in agg.notes:
        agg.notes.append(note_text)
    if node.confidence is not None:
        agg.confidence_samples.append(node.confidence)
    agg.sources.append(source)
    return agg


def build_merge_index(output_dir: Path) -> MergeIndex:
    """Aggregate nodes from validated outputs into a compact index."""
    aggregates: Dict[str, NodeAggregate] = {}

    for json_path in _iter_valid_output_files(output_dir):
        data = _load_output(json_path)
        if data is None:
            continue

        source = SourceMetadata(paper_id=json_path.stem)
        for edge in data.edges:
            agg = _upsert_node_aggregate(aggregates, edge.target_node, source)
            agg.linked_edges.append(_summarize_edge(edge, source))

    return MergeIndex(nodes=aggregates)
