"""Modèles candidats et pipeline d'inférence.

Toutes les variantes partagent la même représentation (TF-IDF mots 1-2 grammes)
pour que la comparaison porte sur le classifieur. On part d'une référence
naïve (classe majoritaire) : un modèle qui ne la bat pas nettement n'a rien appris.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .cleaning import clean_text

# \w+ garde les mots d'une lettre ("t" dans "don't", "5" dans "5 stars") ; aucun mot-outil retiré
TOKEN_PATTERN = r"(?u)\b\w+\b"


def tfidf_words() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),      # bigrammes : "not good", "not great", "never again"
        min_df=3,                # ignore les termes vus dans moins de 3 avis (fautes, bruit)
        max_df=0.9,
        sublinear_tf=True,
        token_pattern=TOKEN_PATTERN,
        dtype=np.float32,
    )


def candidates(seed: int) -> dict[str, Pipeline]:
    def pipe(clf):
        return Pipeline([("tfidf", tfidf_words()), ("clf", clf)])

    return {
        "baseline_classe_majoritaire": Pipeline([("tfidf", tfidf_words()),
                                                 ("clf", DummyClassifier(strategy="most_frequent"))]),
        "naive_bayes": pipe(ComplementNB(alpha=0.5)),
        "logreg_C1": pipe(LogisticRegression(C=1, max_iter=2000, random_state=seed)),
        "logreg_C4": pipe(LogisticRegression(C=4, max_iter=2000, random_state=seed)),
        "logreg_C10": pipe(LogisticRegression(C=10, max_iter=2000, random_state=seed)),
        # classes ~43 % / 57 % : on teste si rééquilibrer les poids aide la classe négative
        "logreg_C4_equilibre": pipe(LogisticRegression(C=4, class_weight="balanced", max_iter=2000,
                                                       random_state=seed)),
        "svm_lineaire_C0.1": pipe(LinearSVC(C=0.1, random_state=seed)),
        "svm_lineaire_C0.5": pipe(LinearSVC(C=0.5, random_state=seed)),
    }


class TextCleaner(BaseEstimator, TransformerMixin):
    """Applique exactement le nettoyage de l'entraînement avant la prédiction.
    Embarqué dans le modèle sauvegardé : impossible d'oublier le prétraitement
    en production (cause classique d'écart entraînement / production)."""

    def __init__(self, cleaning_cfg: dict):
        self.cleaning_cfg = cleaning_cfg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).map(lambda s: clean_text(s, self.cleaning_cfg)).tolist()


def inference_pipeline(fitted_model: Pipeline, cleaning_cfg: dict) -> Pipeline:
    return Pipeline([("clean", TextCleaner(cleaning_cfg)), ("model", fitted_model)])
