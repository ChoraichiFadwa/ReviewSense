"""Étape MÉTIER : passer d'une prédiction par avis à un indicateur par entreprise.

Le besoin exprimé n'est pas « classer des avis » mais « analyser la réputation
des compagnies ». On agrège donc les prédictions par entreprise, puis par secteur :

- part d'avis positifs prédits ;
- borne basse de Wilson (95 %) : une entreprise avec 3 avis positifs sur 3
  n'est pas « meilleure » qu'une entreprise à 90 % sur 200 avis ; la borne basse
  pénalise les petits volumes et permet un classement honnête ;
- on ne publie un score qu'à partir de `min_reviews` avis.

On vérifie aussi que le score prédit suit le score réel (corrélation, écart
moyen), ce qui valide l'usage du modèle pour CE besoin et pas seulement sa
précision avis par avis. Les entreprises du jeu de test n'ont jamais été vues
à l'entraînement : c'est la situation d'une nouvelle entreprise à analyser.

ATTENTION (constat du rapport qualité) : la source contient au plus 20 avis par
niveau d'étoiles et par entreprise. La part d'avis positifs calculée ici est donc
celle de l'ÉCHANTILLON, pas la vraie réputation : de nombreuses entreprises
ressortent à exactement 50 %. La méthode est validée ; les valeurs absolues ne
doivent pas être interprétées comme la réputation réelle.
"""
from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ensure_parent, load_config
from .evaluate import GRID, INK, INK2, NEG, POS, SURFACE, _style

log = logging.getLogger("reputation")


def wilson_lower_bound(pos: pd.Series, n: pd.Series, z: float = 1.96) -> pd.Series:
    p = pos / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centre - margin) / denom


def _agg(g: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    agg = g.agg(
        nb_avis=("review_id", "size"),
        nb_positifs_predits=("pos_pred", "sum"),
        part_positifs_predite=("pos_pred", "mean"),
        part_positifs_reelle=("pos_true", "mean"),
        proba_positif_moyenne=("proba_positif", "mean"),
    ).reset_index()
    agg["score_reputation_wilson"] = wilson_lower_bound(agg["nb_positifs_predits"], agg["nb_avis"])
    agg["ecart_predit_reel"] = agg["part_positifs_predite"] - agg["part_positifs_reelle"]
    return agg


def aggregate(pred: pd.DataFrame, entities: pd.DataFrame, min_reviews: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pred.assign(pos_pred=(pred["pred"] == 1).astype(int), pos_true=(pred["label"] == 1).astype(int))
    g = g.merge(entities, on="entity_id", how="left")
    by_company = _agg(g.groupby(["entity_id", "entity_category"], dropna=False))
    by_company["publiable"] = by_company["nb_avis"] >= min_reviews
    by_company = by_company.sort_values("score_reputation_wilson", ascending=False).reset_index(drop=True)
    by_category = _agg(g.groupby("entity_category", dropna=False))
    by_category = by_category.sort_values("part_positifs_predite", ascending=False).reset_index(drop=True)
    return by_company, by_category


def plot_pred_vs_true(agg: pd.DataFrame, path):
    d = agg[agg["publiable"]]
    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1, zorder=1)
    ax.scatter(d["part_positifs_reelle"], d["part_positifs_predite"], s=26, color=POS, alpha=0.7,
               edgecolor=SURFACE, linewidth=0.8, zorder=2)
    r = d["part_positifs_reelle"].corr(d["part_positifs_predite"])
    mae = d["ecart_predit_reel"].abs().mean() * 100
    ax.text(0.03, 0.97, f"corrélation r = {r:.3f}\nécart moyen = {mae:.1f} points\n{len(d)} entreprises",
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax.set_xlabel("Part d'avis positifs – réelle (étoiles)", color=INK2)
    ax.set_ylabel("Part d'avis positifs – prédite (modèle)", color=INK2)
    ax.set_title("Réputation par entreprise : prédit vs réel", fontsize=11, loc="left", color=INK)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=150)
    plt.close(fig)


def plot_ranking(agg: pd.DataFrame, path, n: int = 8):
    d = agg[agg["publiable"]]
    top, bottom = d.head(n), d.tail(n)
    sel = pd.concat([top, bottom]).drop_duplicates("entity_id").sort_values("score_reputation_wilson")
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(sel) + 1.2), facecolor=SURFACE)
    colors = [POS if s >= 0.5 else NEG for s in sel["part_positifs_predite"]]
    ax.barh(sel["entity_id"], sel["score_reputation_wilson"] * 100, color=colors, height=0.6)
    for y, (v, nb) in enumerate(zip(sel["score_reputation_wilson"], sel["nb_avis"])):
        ax.text(v * 100 + 1, y, f"{v * 100:.0f}  ({nb} avis)", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Score de réputation = borne basse de Wilson de la part d'avis positifs prédits (%)",
                  color=INK2, fontsize=9)
    ax.set_title("Classement sur l'échantillon de test (non représentatif : max 20 avis par note)",
                 fontsize=11, loc="left", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=150)
    plt.close(fig)


def run(cfg: dict) -> dict:
    out = cfg["paths"]["outputs_dir"]
    pred = pd.read_parquet(out / "test_predictions.parquet")
    clean = pd.read_parquet(cfg["paths"]["processed"])
    ent_cols = ["entity_id"] + [c for c in clean.columns if c == "entity_category"]
    entities = clean[ent_cols].drop_duplicates("entity_id")
    if "entity_category" not in entities:
        entities["entity_category"] = "non renseigné"

    by_company, by_category = aggregate(pred, entities, int(cfg["reputation"]["min_reviews"]))
    by_company.to_csv(ensure_parent(out / "reputation_by_company.csv"), index=False, encoding="utf-8")
    by_category.to_csv(out / "reputation_by_category.csv", index=False, encoding="utf-8")
    plot_pred_vs_true(by_company, out / "figures" / "reputation_pred_vs_true.png")
    plot_ranking(by_company, out / "figures" / "reputation_ranking.png")

    pub = by_company[by_company["publiable"]]
    summary = {
        "entreprises_test": int(len(by_company)),
        "entreprises_publiables": int(len(pub)),
        "seuil_avis": int(cfg["reputation"]["min_reviews"]),
        "correlation_predit_reel": float(pub["part_positifs_reelle"].corr(pub["part_positifs_predite"])),
        "ecart_absolu_moyen_points": float(pub["ecart_predit_reel"].abs().mean() * 100),
        "ecart_absolu_max_points": float(pub["ecart_predit_reel"].abs().max() * 100),
        "secteurs": int(len(by_category)),
    }
    with open(out / "reputation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Réputation : %d entreprises publiables, corrélation prédit/réel = %.3f, écart moyen = %.1f pts",
             summary["entreprises_publiables"], summary["correlation_predit_reel"],
             summary["ecart_absolu_moyen_points"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
