"""Métriques, graphiques et analyse d'erreurs."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from .config import ensure_parent

# Palette (validée daltonisme) : négatif = rouge, positif = bleu, neutre = gris
NEG, POS, GREY = "#e34948", "#2a78d6", "#898781"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def scores(model, X) -> np.ndarray:
    """Score continu (probabilité si disponible) pour l'AUC et les seuils."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, list(model.classes_).index(1)]
        except AttributeError:
            pass
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[-1, 1], zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_negatif": p[0], "rappel_negatif": r[0], "f1_negatif": f[0],
        "precision_positif": p[1], "rappel_positif": r[1], "f1_positif": f[1],
    }
    if y_score is not None and len(np.unique(y_score)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["roc_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.title.set_color(INK)


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.6, 4), facecolor=SURFACE)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1f} %".replace(",", " "), ha="center", va="center",
                    color="white" if pct[i, j] > 60 else INK, fontsize=11)
    ax.set_xticks([0, 1], ["Négatif (-1)", "Positif (1)"])
    ax.set_yticks([0, 1], ["Négatif (-1)", "Positif (1)"])
    ax.set_xlabel("Prédit", color=INK2)
    ax.set_ylabel("Réel", color=INK2)
    ax.set_title("Matrice de confusion – jeu de test", fontsize=11, loc="left")
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=150)
    plt.close(fig)


def plot_model_comparison(comp: pd.DataFrame, chosen: str, path):
    d = comp.sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    colors = [POS if m == chosen else GREY for m in d["modele"]]
    ax.barh(d["modele"], d["f1_macro"], color=colors, height=0.6)
    for y, v in enumerate(d["f1_macro"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 macro (validation)", color=INK2)
    ax.set_title("Comparaison des modèles – en bleu : modèle retenu", fontsize=11, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=150)
    plt.close(fig)


def top_features(model, n: int = 15) -> pd.DataFrame:
    """Termes les plus associés à chaque polarité (coefficients de la régression logistique)."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame()
    names = np.asarray(vec.get_feature_names_out())
    coef = clf.coef_.ravel() * (1 if clf.classes_[1] == 1 else -1)
    order = np.argsort(coef)
    neg = pd.DataFrame({"terme": names[order[:n]], "poids": coef[order[:n]], "polarite": "négatif"})
    pos = pd.DataFrame({"terme": names[order[-n:]][::-1], "poids": coef[order[-n:]][::-1], "polarite": "positif"})
    return pd.concat([pos, neg], ignore_index=True)


def plot_top_features(tf: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), facecolor=SURFACE)
    for ax, pol, color in ((axes[0], "négatif", NEG), (axes[1], "positif", POS)):
        d = tf[tf["polarite"] == pol].copy()
        d["abs"] = d["poids"].abs()
        d = d.sort_values("abs")
        ax.barh(d["terme"], d["abs"], color=color, height=0.6)
        ax.set_title(f"Termes les plus {pol}s", fontsize=11, loc="left")
        ax.set_xlabel("|poids| dans la régression logistique", color=INK2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=150)
    plt.close(fig)


def error_analysis(pred: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Erreurs les plus « sûres d'elles » : les plus instructives à relire."""
    err = pred[pred["label"] != pred["pred"]].copy()
    err["confiance_erreur"] = np.where(err["pred"] == 1, err["proba_positif"], 1 - err["proba_positif"])
    err["type_erreur"] = np.where(err["pred"] == 1, "faux positif", "faux négatif")
    return err.sort_values("confiance_erreur", ascending=False).head(n)[
        ["review_id", "entity_id", "label", "pred", "proba_positif", "confiance_erreur", "type_erreur", "text"]
    ]
