"""Étape MODÈLE : comparaison sur la validation, choix, réentraînement, test.

    python -m src.train [--config config.yaml]

Règle de sélection (fixée AVANT de regarder le test) :
1. meilleur F1 macro sur la validation ;
2. à moins de 0,005 près, on préfère un modèle qui donne une probabilité
   (régression logistique) : la probabilité sert ensuite au score de réputation
   et rend le modèle interprétable.
Le jeu de test n'est utilisé qu'une seule fois, pour le modèle retenu.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import pandas as pd

from .config import ensure_parent, load_config
from .evaluate import (compute_metrics, error_analysis, plot_confusion, plot_model_comparison,
                       plot_top_features, scores, top_features)
from .models import candidates, inference_pipeline
from .split import load_splits

log = logging.getLogger("train")
TOLERANCE = 0.005


def compare(splits: dict, seed: int) -> pd.DataFrame:
    rows = []
    for name, model in candidates(seed).items():
        t0 = time.perf_counter()
        model.fit(splits["train"]["text"], splits["train"]["label"])
        fit_s = time.perf_counter() - t0
        X, y = splits["val"]["text"], splits["val"]["label"]
        m = compute_metrics(y, model.predict(X), scores(model, X))
        rows.append({"modele": name, **m, "probabiliste": hasattr(model, "predict_proba"),
                     "temps_entrainement_s": round(fit_s, 1)})
        log.info("%-28s F1 macro val = %.4f  (%.0f s)", name, m["f1_macro"], fit_s)
    return pd.DataFrame(rows)


def select(comp: pd.DataFrame) -> str:
    best = comp["f1_macro"].max()
    eligible = comp[comp["probabiliste"] & (comp["f1_macro"] >= best - TOLERANCE)]
    pool = eligible if not eligible.empty else comp
    return pool.sort_values("f1_macro", ascending=False).iloc[0]["modele"]


def run(cfg: dict) -> dict:
    seed = cfg["seed"]
    out, models_dir = cfg["paths"]["outputs_dir"], cfg["paths"]["models_dir"]
    splits = load_splits(cfg)

    comp = compare(splits, seed)
    chosen = select(comp)
    comp.to_csv(ensure_parent(out / "model_comparison.csv"), index=False)
    plot_model_comparison(comp, chosen, out / "figures" / "model_comparison.png")
    log.info("Modèle retenu : %s", chosen)

    # Réentraînement sur train + validation, puis évaluation UNIQUE sur le test
    trval = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    model = candidates(seed)[chosen]
    model.fit(trval["text"], trval["label"])

    test = splits["test"]
    pred = test.copy()
    pred["pred"] = model.predict(test["text"])
    pred["proba_positif"] = scores(model, test["text"])
    test_metrics = compute_metrics(test["label"], pred["pred"], pred["proba_positif"])
    log.info("TEST  accuracy = %.4f | F1 macro = %.4f | ROC AUC = %.4f",
             test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["roc_auc"])

    pred.to_parquet(ensure_parent(out / "test_predictions.parquet"), index=False)
    error_analysis(pred).to_csv(out / "error_analysis.csv", index=False, encoding="utf-8")
    plot_confusion(test["label"], pred["pred"], out / "figures" / "confusion_matrix.png")
    tf = top_features(model)
    if not tf.empty:
        tf.to_csv(out / "top_features.csv", index=False, encoding="utf-8")
        plot_top_features(tf, out / "figures" / "top_features.png")

    final = inference_pipeline(model, cfg["cleaning"])
    joblib.dump(final, ensure_parent(models_dir / "sentiment_model.joblib"), compress=3)

    metrics = {
        "modele_retenu": chosen,
        "regle_selection": f"meilleur F1 macro validation ; modèle probabiliste préféré à {TOLERANCE} près",
        "entraine_sur": "train + validation",
        "taille": {k: int(len(v)) for k, v in splits.items()},
        "validation": comp.set_index("modele").loc[chosen, ["accuracy", "f1_macro", "roc_auc"]].to_dict(),
        "test": test_metrics,
    }
    with open(ensure_parent(out / "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
