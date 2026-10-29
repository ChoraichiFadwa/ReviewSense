"""Ingestion : lecture de la source brute et mise au schéma standard.

Schéma standard en sortie (une ligne = un avis) :
    review_id          identifiant stable (hash du contenu source)
    entity_id          entreprise évaluée
    entity_<attribut>  attributs de l'entreprise (secteur, description…)
    title_raw          titre tel que reçu (si la source en a un)
    review_raw         avis tel que reçu
    text_raw           texte analysé = avis, précédé du titre s'il apporte une info
    title_used         True si le titre a été ajouté au texte
    label_raw          note telle que reçue (traçabilité)
    label              1 (positif) / -1 (négatif) ; NaN si exclue ou non reconnue
    label_excluded     True si la note fait partie des notes volontairement écartées
    source_name, source_file, source_row, ingested_at   lignage
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ensure_parent

REQUIRED_LOGICAL_COLUMNS = ("text", "label", "entity")
READERS = {
    "csv": lambda p, s: pd.read_csv(p, sep=s["csv_separator"], encoding=s["encoding"], dtype=str,
                                    keep_default_na=False),
    "jsonl": lambda p, s: pd.read_json(p, lines=True, encoding=s["encoding"], dtype=False),
    "xlsx": lambda p, s: pd.read_excel(p, dtype=str),
    "parquet": lambda p, s: pd.read_parquet(p),
}


class SchemaError(ValueError):
    """La source ne respecte pas le contrat attendu (colonnes manquantes, format inconnu…)."""


def read_source(path: Path, source_cfg: dict) -> pd.DataFrame:
    fmt = source_cfg["format"]
    if fmt not in READERS:
        raise SchemaError(f"Format '{fmt}' non supporté. Formats possibles : {sorted(READERS)}")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {path}. Lancer d'abord scripts/prepare_data.py (voir README)"
        )
    return READERS[fmt](path, source_cfg)


def validate_columns(df: pd.DataFrame, source_cfg: dict) -> None:
    """Contrat d'interface : on échoue tôt et clairement si une colonne manque."""
    columns = source_cfg["columns"]
    missing_logical = [c for c in REQUIRED_LOGICAL_COLUMNS if c not in columns]
    if missing_logical:
        raise SchemaError(f"Correspondance manquante dans config.source.columns : {missing_logical}")
    expected = list(columns.values()) + list((source_cfg.get("entity_attributes") or {}).values())
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"Colonnes absentes de la source : {missing}. Colonnes reçues : {list(df.columns)}")


def _norm_key(v) -> str:
    """'4', '4.0', 4 -> '4' : les notes arrivent souvent en texte ou en flottant selon l'export."""
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    except ValueError:
        return s


def map_labels(raw: pd.Series, mapping: dict) -> pd.Series:
    """Applique la correspondance de notes. Une valeur inconnue devient NaN : elle sera
    rejetée et tracée au nettoyage, pas corrigée en silence."""
    norm = {_norm_key(k): v for k, v in mapping.items()}
    return raw.map(_norm_key).map(norm).astype("float")


def extract_entity(raw: pd.Series, pattern: str | None) -> pd.Series:
    raw = raw.astype(str).str.strip()
    if not pattern:
        return raw
    return raw.str.extract(pattern, expand=False).fillna(raw)


def combine_title_review(title: pd.Series, review: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Ajoute le titre devant l'avis SAUF s'il est redondant.

    Sur Trustpilot, quand l'avis est long, le titre est souvent juste son début
    coupé par « … » : l'ajouter compterait ces mots deux fois.
    """
    t = title.fillna("").astype(str).str.strip()
    r = review.fillna("").astype(str).str.strip()
    stem = t.str.rstrip("…").str.rstrip(".").str.strip()
    redundant = [(not s) or rv.lower().startswith(s.lower()) for s, rv in zip(stem, r)]
    used = ~pd.Series(redundant, index=r.index)
    text = r.where(~used, t + ". " + r)
    return text, used


def _row_hash(*parts: pd.Series) -> pd.Series:
    joined = parts[0].astype(str)
    for p in parts[1:]:
        joined = joined + "\x1f" + p.astype(str)
    return joined.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])


def standardize(df: pd.DataFrame, source_cfg: dict, source_file: Path) -> pd.DataFrame:
    validate_columns(df, source_cfg)
    cols = source_cfg["columns"]

    out = pd.DataFrame({"source_row": range(len(df))}, index=df.index)
    out["entity_id"] = extract_entity(df[cols["entity"]], source_cfg.get("entity_pattern"))
    for name, col in (source_cfg.get("entity_attributes") or {}).items():
        out[f"entity_{name}"] = df[col].astype("string").str.strip()

    out["review_raw"] = df[cols["text"]].astype("string")
    if cols.get("title"):
        out["title_raw"] = df[cols["title"]].astype("string")
        out["text_raw"], out["title_used"] = combine_title_review(out["title_raw"], out["review_raw"])
    else:
        out["title_raw"], out["text_raw"], out["title_used"] = pd.NA, out["review_raw"], False
    out["text_raw"] = out["text_raw"].astype("string")

    out["label_raw"] = df[cols["label"]].astype("string")
    excluded = {_norm_key(v) for v in (source_cfg.get("excluded_labels") or [])}
    out["label_excluded"] = out["label_raw"].map(_norm_key).isin(excluded)
    out["label"] = map_labels(out["label_raw"], source_cfg["label_mapping"])

    # identifiant stable : même avis source -> même id, même si l'ordre des lignes change
    out["review_id"] = _row_hash(out["entity_id"], out["review_raw"].fillna(""), out["label_raw"].fillna(""))
    out["source_name"] = source_cfg.get("name", "source")
    out["source_file"] = Path(source_file).name
    out["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    first = ["review_id", "entity_id"]
    return out[first + [c for c in out.columns if c not in first]].reset_index(drop=True)


def ingest(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (source brute, données au schéma standard)."""
    raw_path = cfg["paths"]["raw"]
    raw = read_source(raw_path, cfg["source"])
    std = standardize(raw, cfg["source"], raw_path)
    std.to_parquet(ensure_parent(cfg["paths"]["interim"]), index=False)
    return raw, std
