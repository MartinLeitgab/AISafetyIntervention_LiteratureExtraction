from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


# ---- Data models ------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    input_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class FalkorDB:
    host: str
    port: int
    graph: str


@dataclass(frozen=True)
class Settings:
    project_root: Path
    paths: Paths
    falkordb: FalkorDB


# ---- Loader -----------------------------------------------------------------

def load_settings(config_path: Path | None = None) -> Settings:
    """
    Load settings from config.yaml.
    Paths in YAML are treated as relative to this file's directory.
    """
    project_root = Path(__file__).resolve().parent
    cfg_file = config_path or (project_root / "config.yaml")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def rel(p: str) -> Path:
        return (project_root / p).resolve()

    # Defaults (in case some keys are missing in YAML)
    paths_cfg = cfg.get("paths", {})
    falkor_cfg = cfg.get("falkordb", {})

    return Settings(
        project_root=project_root,
        paths=Paths(
            input_dir=rel(paths_cfg.get("input_dir", "./intervention_graph_creation/data/raw/pdfs_local")),
            output_dir=rel(paths_cfg.get("output_dir", "./intervention_graph_creation/data/processed")),
        ),
        falkordb=FalkorDB(
            host=falkor_cfg.get("host", "localhost"),
            port=int(falkor_cfg.get("port", 6379)),
            graph=falkor_cfg.get("graph", "AISafetyIntervention"),
        ),
    )
