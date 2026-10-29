"""Chargement de la configuration et résolution des chemins."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict:
    """Charge le YAML et transforme les chemins relatifs en chemins absolus
    (depuis la racine du projet), pour que le code tourne quel que soit le
    dossier depuis lequel on le lance."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {k: _resolve(v) for k, v in cfg["paths"].items()}
    cfg["config_file"] = str(path)
    return cfg


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
