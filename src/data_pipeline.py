"""Étape DATA : ingestion -> profilage -> nettoyage -> profilage -> découpage.

    python -m src.data_pipeline [--config config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import re

from .cleaning import apply_cleaning, save_cleaning_outputs
from .config import load_config
from .ingestion import ingest
from .quality import leakage_between_original_splits, profile, write_report
from .split import save_splits, split_dataset, summarize

log = logging.getLogger("data")


def entity_checks(std) -> dict:
    """Constats sur la qualité des données d'entreprise (dimension)."""
    out = {}
    if "title_used" in std:
        out["titres redondants (début de l'avis tronqué) non ajoutés au texte"] = int(
            std["title_raw"].notna().sum() - std["title_used"].sum())
    per_entity = std.groupby("entity_id").size()
    out["entreprises au plafond de la source (nb d'avis = max observé)"] = int((per_entity == per_entity.max()).sum())
    # Échantillonnage stratifié par note ? (ex. au plus N avis par niveau d'étoiles et par entreprise)
    per_level = std.groupby(["entity_id", "label_raw"]).size()
    cap = int(per_level.max())
    out["maximum d'avis pour une même entreprise et une même note"] = cap
    levels = std["label_raw"].nunique()
    full = per_level[per_level == cap].groupby(level=0).size()
    out[f"entreprises avec exactement {cap} avis pour chacune des {levels} notes"] = int((full == levels).sum())
    for attr in [c for c in std.columns if c.startswith("entity_") and c != "entity_id"]:
        out[f"entreprises avec plusieurs valeurs de « {attr[7:]} »"] = int(
            std.groupby("entity_id")[attr].nunique().gt(1).sum())
    return out


def save_examples(clean, rejected, cfg: dict, n: int = 5) -> None:
    """Petits extraits lisibles (pour le rapport et la démo) : quelques lignes
    en quarantaine par motif, et quelques textes avant / après normalisation."""
    reports = cfg["paths"]["reports_dir"]
    seed = cfg["seed"]
    q = rejected.sample(frac=1, random_state=seed).groupby("reject_reason").head(n).sort_values("reject_reason")
    q[["reject_reason", "label_raw", "language", "text_raw"]].to_csv(
        reports / "quarantine_examples.csv", index=False, encoding="utf-8")
    pattern = r"[,;!?][A-Za-z]|&[a-zA-Z]+;|https?://|(\w)\1{3,}"
    mask = clean["text_raw"].astype(object).map(lambda t: bool(re.search(pattern, t)))
    ex = clean[clean["text_modified"] & mask & (clean["text_raw"].str.len() < 260)]
    ex.sample(min(8, len(ex)), random_state=seed)[["text_raw", "text"]].to_csv(
        reports / "cleaning_examples.csv", index=False, encoding="utf-8")


def run(cfg: dict) -> dict:
    raw, std = ingest(cfg)
    log.info("Ingestion : %s lignes lues depuis %s", f"{len(std):,}", cfg["paths"]["raw"].name)

    ratio = float(cfg["cleaning"]["language_min_ratio"])
    lang = cfg["cleaning"].get("expected_language")
    before = profile(std, "text_raw", lang_ratio=ratio, expected_language=lang)
    clean, rejected, cleaning_log = apply_cleaning(std, cfg)
    save_cleaning_outputs(clean, rejected, cfg)
    after = profile(clean, "text", lang_ratio=ratio, expected_language=lang)
    log.info("Nettoyage : %s lignes conservées, %s en quarantaine",
             f"{len(clean):,}", f"{len(rejected):,}")

    parts = split_dataset(clean, cfg)
    save_splits(parts, cfg)
    split_summary = summarize(parts)

    extra = entity_checks(std)
    if "original_split" in raw.columns:
        tmp = raw.rename(columns={cfg["source"]["columns"]["text"]: "_text"})
        extra["textes présents dans plusieurs splits du découpage d'origine (fuite)"] = \
            leakage_between_original_splits(tmp, "original_split", "_text")

    rejected_counts = {str(k): int(v) for k, v in rejected["reject_reason"].value_counts().items()}
    write_report(before, after, cleaning_log, split_summary, rejected_counts, extra, cfg)
    save_examples(clean, rejected, cfg)
    log.info("Rapport qualité : %s", cfg["paths"]["reports_dir"] / "data_quality_report.md")
    return {"before": before, "after": after, "split": split_summary, "rejected": rejected_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
