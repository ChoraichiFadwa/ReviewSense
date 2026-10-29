"""Étape BI : exposer les résultats dans un petit datamart (SQLite) en étoile.

    dim_entreprise          une ligne par entreprise : secteur, description
    fait_avis               une ligne par avis : note source, label réel, prédiction, probabilité
    v_reputation_entreprise vue agrégée par entreprise (prête pour Power BI, Metabase, SQL…)
    v_reputation_secteur    vue agrégée par secteur

La description de l'entreprise était répétée sur chaque avis dans la source :
elle est rangée une seule fois dans la dimension.
On ne charge que les avis du jeu de test (entreprises jamais vues à
l'entraînement) : des prédictions faites sur les données d'entraînement
seraient trop optimistes. En production, on y chargerait les nouveaux avis.
SQLite suffit pour la démonstration ; la même structure irait dans
l'entrepôt de données (PostgreSQL, SQL Server…).
"""
from __future__ import annotations

import argparse
import logging
import sqlite3

import pandas as pd

from .config import ensure_parent, load_config

log = logging.getLogger("datamart")

DDL = """
DROP VIEW IF EXISTS v_reputation_entreprise;
DROP VIEW IF EXISTS v_reputation_secteur;
DROP TABLE IF EXISTS fait_avis;
DROP TABLE IF EXISTS dim_entreprise;

CREATE TABLE dim_entreprise (
    entity_id        TEXT PRIMARY KEY,
    secteur          TEXT,
    description      TEXT,
    source_name      TEXT NOT NULL
);

CREATE TABLE fait_avis (
    review_id        TEXT PRIMARY KEY,
    entity_id        TEXT NOT NULL REFERENCES dim_entreprise(entity_id),
    split            TEXT NOT NULL,
    note_source      INTEGER,
    label_reel       INTEGER NOT NULL CHECK (label_reel IN (-1, 1)),
    label_predit     INTEGER NOT NULL CHECK (label_predit IN (-1, 1)),
    proba_positif    REAL NOT NULL CHECK (proba_positif BETWEEN 0 AND 1),
    longueur_texte   INTEGER NOT NULL
);
CREATE INDEX ix_fait_avis_entity ON fait_avis(entity_id);

CREATE VIEW v_reputation_entreprise AS
SELECT  e.entity_id,
        e.secteur,
        COUNT(*)                                                         AS nb_avis,
        ROUND(AVG(CASE WHEN f.label_predit = 1 THEN 1.0 ELSE 0 END), 4)  AS part_positifs_predite,
        ROUND(AVG(CASE WHEN f.label_reel   = 1 THEN 1.0 ELSE 0 END), 4)  AS part_positifs_reelle,
        ROUND(AVG(f.proba_positif), 4)                                   AS proba_positif_moyenne,
        ROUND(AVG(f.note_source), 2)                                     AS note_moyenne_source
FROM fait_avis f
JOIN dim_entreprise e ON e.entity_id = f.entity_id
GROUP BY e.entity_id, e.secteur;

CREATE VIEW v_reputation_secteur AS
SELECT  e.secteur,
        COUNT(DISTINCT e.entity_id)                                      AS nb_entreprises,
        COUNT(*)                                                         AS nb_avis,
        ROUND(AVG(CASE WHEN f.label_predit = 1 THEN 1.0 ELSE 0 END), 4)  AS part_positifs_predite,
        ROUND(AVG(CASE WHEN f.label_reel   = 1 THEN 1.0 ELSE 0 END), 4)  AS part_positifs_reelle
FROM fait_avis f
JOIN dim_entreprise e ON e.entity_id = f.entity_id
GROUP BY e.secteur;
"""


def run(cfg: dict) -> None:
    out = cfg["paths"]["outputs_dir"]
    pred = pd.read_parquet(out / "test_predictions.parquet")
    clean = pd.read_parquet(cfg["paths"]["processed"])

    fact = pd.DataFrame({
        "review_id": pred["review_id"],
        "entity_id": pred["entity_id"].astype(str),
        "split": "test",
        "note_source": pd.to_numeric(pred["label_raw"], errors="coerce").astype("Int64"),
        "label_reel": pred["label"].astype(int),
        "label_predit": pred["pred"].astype(int),
        "proba_positif": pred["proba_positif"].astype(float),
        "longueur_texte": pred["text"].str.len().astype(int),
    })
    ent = clean.drop_duplicates("entity_id").set_index("entity_id")
    ids = pd.Index(fact["entity_id"].unique())
    dim = pd.DataFrame({
        "entity_id": ids,
        "secteur": ent["entity_category"].reindex(ids).values if "entity_category" in ent else None,
        "description": ent["entity_description"].reindex(ids).values if "entity_description" in ent else None,
        "source_name": cfg["source"]["name"],
    })

    with sqlite3.connect(ensure_parent(cfg["paths"]["datamart"])) as con:
        con.executescript(DDL)
        dim.to_sql("dim_entreprise", con, if_exists="append", index=False)
        fact.to_sql("fait_avis", con, if_exists="append", index=False)
        # contrôles de cohérence après chargement (recette technique)
        n_fact = con.execute("SELECT COUNT(*) FROM fait_avis").fetchone()[0]
        orphans = con.execute(
            "SELECT COUNT(*) FROM fait_avis f LEFT JOIN dim_entreprise e USING(entity_id) "
            "WHERE e.entity_id IS NULL").fetchone()[0]
        n_view = con.execute("SELECT SUM(nb_avis) FROM v_reputation_entreprise").fetchone()[0]
    assert n_fact == len(fact), "Le nombre de lignes chargées ne correspond pas à la source"
    assert orphans == 0, "Des avis référencent une entreprise absente de la dimension"
    assert n_view == n_fact, "La vue agrégée ne retrouve pas tous les avis"
    log.info("Datamart : %d avis, %d entreprises -> %s", n_fact, len(dim), cfg["paths"]["datamart"].name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
