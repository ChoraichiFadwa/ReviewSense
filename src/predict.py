"""Prédire la polarité de nouveaux avis avec le modèle entraîné.

    python -m src.predict "Service client injoignable, je déconseille."
    python -m src.predict --input nouveaux_avis.csv --text-col avis --output predictions.csv

Le modèle sauvegardé embarque le nettoyage du texte : on lui donne le texte brut.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config


def load_model(cfg: dict):
    path = cfg["paths"]["models_dir"] / "sentiment_model.joblib"
    if not path.exists():
        raise SystemExit(f"Modèle introuvable ({path}). Lancer d'abord : python -m src.pipeline")
    return joblib.load(path)


def predict(model, texts: list[str]) -> pd.DataFrame:
    proba = model.predict_proba(texts)[:, list(model.classes_).index(1)]
    return pd.DataFrame({"texte": texts, "polarite": model.predict(texts), "proba_positif": proba.round(3)})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("texts", nargs="*", help="un ou plusieurs avis")
    p.add_argument("--input", help="fichier CSV d'avis")
    p.add_argument("--text-col", default="text")
    p.add_argument("--output", help="fichier CSV de sortie")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    model = load_model(load_config(args.config))
    if args.input:
        df = pd.read_csv(args.input)
        if args.text_col not in df.columns:
            raise SystemExit(f"Colonne '{args.text_col}' absente. Colonnes : {list(df.columns)}")
        res = predict(model, df[args.text_col].fillna("").astype(str).tolist())
        df["polarite"], df["proba_positif"] = res["polarite"].values, res["proba_positif"].values
        out = Path(args.output or Path(args.input).with_name(Path(args.input).stem + "_predictions.csv"))
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"{len(df)} avis prédits -> {out}")
    elif args.texts:
        with pd.option_context("display.max_colwidth", 80, "display.width", 120):
            print(predict(model, args.texts).to_string(index=False))
    else:
        p.error("donner au moins un avis, ou --input")


if __name__ == "__main__":
    main()
