from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ---- Data models ------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    extraction_error_dir: Path
    graph_error_dir: Path
    embeddings_error_dir: Path


@dataclass(frozen=True)
class FalkorDB:
    host: str
    port: int
    graph: str


@dataclass(frozen=True)
class Embeddings:
    model: str
    batch_size: int
    max_cuncurrent_batches: int
    type: str  # "narrow" | "full"


@dataclass(frozen=True)
class Extraction:
    total_articles: int
    batch_size: int


@dataclass(frozen=True)
class Settings:
    project_root: Path
    paths: Paths
    falkordb: FalkorDB
    embeddings: Embeddings
    extraction: Extraction


# ---- Loader -----------------------------------------------------------------


def load_settings(config_path: Path | None = None) -> Settings:
    project_root = Path(__file__).resolve().parent
    cfg_file = config_path or (project_root / "config.yaml")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def rel(p: str) -> Path:
        return (project_root / p).resolve()

    paths_cfg = cfg.get("paths", {})
    falkor_cfg = cfg.get("falkordb", {})
    emb_cfg = cfg.get("embeddings", {})
    extraction_cfg = cfg.get("extraction", {})

    output_dir = rel(
        paths_cfg.get("output_dir", "./intervention_graph_creation/data/processed")
    )
    extraction_error_sub = paths_cfg.get("extraction_error_dir", "./extraction_error")
    extraction_error_dir = (output_dir / extraction_error_sub).resolve()
    graph_error_sub = paths_cfg.get("graph_error_dir", "./graph_error")
    graph_error_dir = (output_dir / graph_error_sub).resolve()
    embeddings_error_sub = paths_cfg.get("embeddings_error_dir", "./embeddings_error")
    embeddings_error_dir = (output_dir / embeddings_error_sub).resolve()

    return Settings(
        project_root=project_root,
        paths=Paths(
            input_dir=rel(
                paths_cfg.get(
                    "input_dir", "./intervention_graph_creation/data/raw/pdfs_local"
                )
            ),
            output_dir=output_dir,
            logs_dir=rel(paths_cfg.get("logs_dir", "./logs")),
            extraction_error_dir=extraction_error_dir,
            graph_error_dir=graph_error_dir,
            embeddings_error_dir=embeddings_error_dir,
        ),
        falkordb=FalkorDB(
            host=falkor_cfg.get("host", "localhost"),
            port=int(falkor_cfg.get("port", 6379)),
            graph=falkor_cfg.get("graph", "AISafetyIntervention"),
        ),
        embeddings=Embeddings(
            model=emb_cfg.get("model", "text-embedding-3-large"),
            batch_size=int(emb_cfg.get("batch_size", 256)),
            max_cuncurrent_batches=int(emb_cfg.get("max_cuncurrent_batches", 120)),
            type=emb_cfg.get("type", "narrow"),
        ),
        extraction=Extraction(
            total_articles=int(extraction_cfg.get("total_articles", 20)),
            batch_size=int(extraction_cfg.get("batch_size", 5)),
        ),
    )
