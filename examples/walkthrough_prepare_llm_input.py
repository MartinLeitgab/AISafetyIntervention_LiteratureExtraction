#!/usr/bin/env python3
"""Walkthrough: Prepare data-only LLM inputs for node comparison (Step 1)

This script demonstrates how to:
1) Read structured extraction outputs from the `output/` directory
2) Build an in-memory merge index of nodes with aliases, context, and sources
3) Produce a data-only payload for comparing two nodes (no prompt text)

Usage examples:
  uv run python examples/walkthrough_prepare_llm_input.py
  uv run python examples/walkthrough_prepare_llm_input.py --key-a <node_key_a> --key-b <node_key_b>
  uv run python examples/walkthrough_prepare_llm_input.py --save output/node_comparison_example.json

The resulting payload contains exactly the fields required for LLM-assisted merge decisions,
without including any task instructions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

# Ensure project root is on sys.path so `src` is importable when running from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.merge_indexer import build_merge_index  # noqa: E402
from src.merge_input_builder import build_node_comparison_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a data-only comparison payload for two nodes from extraction outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing structured OutputSchema JSON files (not raw responses).",
    )
    parser.add_argument(
        "--key-a",
        type=str,
        default=None,
        help="Node key for the first node (defaults to the first key found).",
    )
    parser.add_argument(
        "--key-b",
        type=str,
        default=None,
        help="Node key for the second node (defaults to the second key found).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the first N available node keys and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many keys to list when using --list.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the generated comparison payload as JSON.",
    )
    return parser.parse_args()


def select_keys(
    index_keys: list[str], key_a: Optional[str], key_b: Optional[str]
) -> tuple[str, str]:
    if key_a and key_b:
        return key_a, key_b
    if len(index_keys) < 2:
        raise RuntimeError("Need at least two nodes to build a comparison payload.")
    return index_keys[0], index_keys[1]


def main() -> None:
    args = parse_args()

    if not args.output_dir.exists():
        raise FileNotFoundError(
            f"Output directory not found: {args.output_dir}. Run the extractor first."
        )

    index = build_merge_index(args.output_dir)
    node_keys = list(index.nodes.keys())

    if args.list:
        print(f"Total nodes: {len(node_keys)}")
        print("First keys:")
        for k in node_keys[: max(0, args.limit)]:
            print(f" - {k}")
        return

    key_a, key_b = select_keys(node_keys, args.key_a, args.key_b)

    payload = build_node_comparison_input(index, key_a, key_b)
    payload_json = payload.model_dump_json(indent=2)

    print(f"Total nodes indexed: {len(node_keys)}")
    print(f"Node A key: {key_a}")
    print(f"Node B key: {key_b}")
    print("\nData-only comparison payload:\n")
    print(payload_json)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("w", encoding="utf-8") as fh:
            fh.write(payload_json)
        print(f"\nSaved payload to: {args.save}")


if __name__ == "__main__":
    main()
