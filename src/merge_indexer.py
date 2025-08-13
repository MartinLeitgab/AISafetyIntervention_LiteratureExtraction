from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic import BaseModel, ValidationError

try:
    from src.prompts import OutputSchema
except ImportError:
    from prompts import OutputSchema

from .merge_types import NodeAggregate, LinkedEdgeSummary, SourceMetadata


class MergeIndex(BaseModel):
    nodes: Dict[str, NodeAggregate]


def _node_key(canonical_name: str) -> str:
    return canonical_name.strip().lower()


def build_merge_index(output_dir: Path) -> MergeIndex:
    """Read parsed output JSON files and aggregate nodes.

    output_dir should contain structured OutputSchema JSON files (not the raw_response files).
    Non-matching JSON files will be safely skipped.
    """
    aggregates: Dict[str, NodeAggregate] = {}

    json_paths = sorted(
        [p for p in output_dir.glob("*.json") if "raw_response" not in p.name]
    )

    for json_path in json_paths:
        try:
            data = OutputSchema.model_validate_json(
                json_path.read_text(encoding="utf-8")
            )
        except (ValidationError, ValueError):
            # Skip files that are not OutputSchema (e.g., example payloads saved in output/)
            continue

        paper_id = json_path.stem
        source = SourceMetadata(paper_id=paper_id)

        for edge in data.edges:
            node = edge.target_node
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
            else:
                agg = aggregates[key]
                # Update primary text to the most recent surface form if canonical matches; keep first otherwise
                if (
                    node.canonical_name
                    and node.canonical_name.lower() == agg.canonical_text.lower()
                ):
                    agg.text = node.name
                # Merge aliases and notes
                for alias in node.aliases or []:
                    if alias not in agg.aliases:
                        agg.aliases.append(alias)
                note_text = getattr(node, "notes", None)
                if note_text and note_text not in agg.notes:
                    agg.notes.append(note_text)
                if node.confidence is not None:
                    agg.confidence_samples.append(node.confidence)
                agg.sources.append(source)

            # Record the connecting edge as context
            aggregates[key].linked_edges.append(
                LinkedEdgeSummary(
                    edge_type=edge.type,
                    rationale=edge.rationale,
                    confidence=edge.confidence,
                    source=source,
                )
            )

    return MergeIndex(nodes=aggregates)
