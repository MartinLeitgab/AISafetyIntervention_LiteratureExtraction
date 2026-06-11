# pyright: standard
from pathlib import Path


from extraction_validator.judge import JudgeInput, KGJudge
from extraction_validator.utilities import DataSource, OpenAICompletionsRequest
from intervention_graph_creation.src.local_graph_extraction.core.edge import GraphEdge
from intervention_graph_creation.src.local_graph_extraction.core.node import GraphNode
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import (
    Meta,
    PaperSchema,
)
from extraction_validator.schema import ProposedFixes


def _minimal_kg() -> PaperSchema:
    return PaperSchema(
        nodes=[
            GraphNode(
                name="Unintended consequences",
                type="concept",
                description="Negative side effects from AI interventions.",
                aliases=["side effects", "unintended effects"],
                concept_category="risk",
            ),
            GraphNode(
                name="Red-teaming",
                type="intervention",
                description="Structured adversarial testing to find model vulnerabilities.",
                aliases=["adversarial evaluation", "red teaming"],
                intervention_lifecycle=3,
                intervention_maturity=2,
            ),
        ],
        edges=[
            GraphEdge(
                source_node="Red-teaming",
                target_node="Unintended consequences",
                type="mitigates",
                description="Red-teaming surfaces risks before deployment.",
                edge_confidence=4,
            )
        ],
        meta=[Meta(key="url", value="https://example.com/paper")],
    )


def _minimal_judge_input() -> JudgeInput:
    return JudgeInput(
        original_text="Sample source text about AI safety interventions.",
        kg_output=_minimal_kg(),
        data_source=DataSource(
            url="https://example.com/paper",
            paper_id="paper123",
            ard_file_source="arxiv",
        ),
    )


# ---------------------------------------------------------------------------
# Bug 1 — OpenAI request must use gpt-5-nano, not any deprecated preview model
# ---------------------------------------------------------------------------


def test_openai_request_uses_current_model(tmp_path: Path) -> None:
    judge = KGJudge(type="OpenAI", output_dir=tmp_path)
    requests = judge._build_requests([_minimal_judge_input()])

    assert len(requests) == 1
    req = requests[0]
    assert isinstance(req.request, OpenAICompletionsRequest)

    model: str = req.request.request["body"]["model"]
    assert model == "gpt-5-nano", f"Expected gpt-5-nano, got {model!r}"
    assert "vision" not in model, (
        f"Model must not be a vision-preview variant: {model!r}"
    )


# ---------------------------------------------------------------------------
# Shared helpers for bugs 2-5
# ---------------------------------------------------------------------------


def _concept_fix(name: str, concept_category: str) -> dict:
    return {
        "new_node": {
            "name": name,
            "type": "concept",
            "description": f"Test concept node: {name}.",
            "aliases": [name.lower(), f"alias {name.lower()}"],
            "concept_category": concept_category,
        },
        "new_edges": [],
    }


def _intervention_fix(name: str) -> dict:
    return {
        "new_node": {
            "name": name,
            "type": "intervention",
            "description": f"Test intervention node: {name}.",
            "aliases": [name.lower(), f"alias {name.lower()}"],
            "intervention_lifecycle": 2,
            "intervention_maturity": 3,
        },
        "new_edges": [],
    }


# ---------------------------------------------------------------------------
# Bug 2 — concept_category must come from the response, not be hardcoded
# ---------------------------------------------------------------------------


def test_concept_category_comes_from_response(tmp_path: Path) -> None:
    judge = KGJudge(type="Anthropic", output_dir=tmp_path)
    fixes = ProposedFixes.model_validate(
        {"add_nodes": [_concept_fix("Node A", "theoretical insight")]}
    )

    result = judge._apply_fixes_to_graph(_minimal_kg(), fixes, "sample text")
    nodes = {n["name"]: n for n in result["nodes"]}

    assert nodes["Node A"]["concept_category"] == "theoretical insight", (
        f"Bug 2: expected 'theoretical insight', got {nodes['Node A']['concept_category']!r}"
    )


# ---------------------------------------------------------------------------
# Bug 3 — intervention aliases must be a flat List[str], not List[List[str]]
# ---------------------------------------------------------------------------


def test_intervention_aliases_are_flat_list(tmp_path: Path) -> None:
    judge = KGJudge(type="Anthropic", output_dir=tmp_path)
    fixes = ProposedFixes.model_validate({"add_nodes": [_intervention_fix("Node B")]})

    result = judge._apply_fixes_to_graph(_minimal_kg(), fixes, "sample text")
    nodes = {n["name"]: n for n in result["nodes"]}

    aliases = nodes["Node B"]["aliases"]
    assert isinstance(aliases, list), "Bug 3: aliases should be a list"
    assert all(isinstance(a, str) for a in aliases), (
        f"Bug 3: aliases must be List[str], not List[List[str]]: {aliases!r}"
    )


# ---------------------------------------------------------------------------
# Bug 4 — intervention nodes must have concept_category=None
# ---------------------------------------------------------------------------


def test_intervention_concept_category_is_none(tmp_path: Path) -> None:
    judge = KGJudge(type="Anthropic", output_dir=tmp_path)
    fixes = ProposedFixes.model_validate({"add_nodes": [_intervention_fix("Node C")]})

    result = judge._apply_fixes_to_graph(_minimal_kg(), fixes, "sample text")
    nodes = {n["name"]: n for n in result["nodes"]}

    assert nodes["Node C"]["concept_category"] is None, (
        f"Bug 4: intervention concept_category must be None, got {nodes['Node C']['concept_category']!r}"
    )


# ---------------------------------------------------------------------------
# Bug 5 — all add_nodes must be processed (no early break after the first)
# ---------------------------------------------------------------------------


def test_all_add_nodes_are_processed(tmp_path: Path) -> None:
    judge = KGJudge(type="Anthropic", output_dir=tmp_path)
    fixes = ProposedFixes.model_validate(
        {
            "add_nodes": [
                _concept_fix("Node D", "risk"),
                _intervention_fix("Node E"),
                _concept_fix("Node F", "design rationale"),
            ]
        }
    )

    result = judge._apply_fixes_to_graph(_minimal_kg(), fixes, "sample text")
    node_names = {n["name"] for n in result["nodes"]}

    assert "Node D" in node_names, (
        "Bug 5: Node D missing — loop broke before processing all nodes"
    )
    assert "Node E" in node_names, (
        "Bug 5: Node E missing — loop broke before processing all nodes"
    )
    assert "Node F" in node_names, "Bug 5: Node F missing — loop broke after first node"
