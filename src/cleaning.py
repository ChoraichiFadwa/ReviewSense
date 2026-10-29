"""Nettoyage des textes et règles de rejet.

Principe : chaque transformation est une règle nommée et chaque ligne écartée
est envoyée en quarantaine avec un motif (`reject_reason`). Rien n'est supprimé
silencieusement : on peut toujours expliquer combien de lignes ont été perdues
et pourquoi.

Choix délibérés :
- on NE supprime PAS les mots-outils : "not", "never", "no" portent la polarité
  ("not good" ≠ "good") ;
- on NE supprime PAS les avis courts : "Great service" ou "Never again" sont des
  avis valides et très informatifs ;
- on garde la casse et les émojis dans le texte nettoyé (lisible pour l'analyse
  d'erreurs) ; la mise en minuscules est faite par le vectoriseur.
"""
from __future__ import annotations

import html
import re
import unicodedata

import ftfy
import pandas as pd

from .config import ensure_parent

RE_HTML_TAG = re.compile(r"<[a-zA-Z/][^>]{0,200}>")
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
RE_INVISIBLE = re.compile("[​‌‍⁠﻿]")
RE_SPACES = re.compile(r"[\s  ]+")
# ponctuation collée au mot suivant : "service,very" / "brand.My dog"
RE_PUNCT_NO_SPACE = re.compile(r"([,;:!?])(?=[A-Za-zÀ-ÿ])")
RE_DOT_NO_SPACE = re.compile(r"(?<=[a-zà-ÿ.])\.(?=[A-ZÀ-Ý][a-zà-ÿ])")
RE_REPEATED_PUNCT = re.compile(r"([!?.¡])\1{3,}")
RE_NON_WORD = re.compile(r"[^\w]+")
RE_WORD = re.compile(r"[a-zà-ÿ']+")
RE_HAS_LETTER = re.compile(r"[^\W\d_]")
RE_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ÿ]")

# Mots-outils fréquents, utilisés uniquement pour l'heuristique de langue.
# Les mots communs à l'anglais sont volontairement exclus des autres listes
# ("on", "me", "son", "plus", "per", "con", "ma"… : "delivered on time" n'est pas du français).
FUNCTION_WORDS = {
    "en": frozenset(
        "the and is was this that with for of to it but very not are have you they "
        "i it's don't my we were be had would on so at".split()),
    "fr": frozenset(
        "le la les un une des et est du ce cette ces pas que qui avec pour dans sur mais "
        "très bien bon je elle nous vous ils au aux ne tout sans aucun "
        "après avoir trop vraiment sa ses mon ma mes se à où c'est j'ai".split()),
    "es": frozenset(
        "el los las y es muy por para pero estoy todo bien gracias del lo".split()),
    "it": frozenset(
        "il gli della di che è molto sono anche una ottimo grazie".split()),
}


def clean_text(text: str, cfg: dict) -> str:
    """Normalise un avis. Utilisée à l'entraînement ET à la prédiction
    (même transformation des deux côtés, sinon le modèle voit des données différentes)."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or text is pd.NA:
        return ""
    t = str(text)
    t = ftfy.fix_text(t)                     # mojibake : "â€™" -> "’"
    t = html.unescape(t)                     # "&quot;" -> '"'
    t = RE_HTML_TAG.sub(" ", t)              # "<br/>" -> " "
    t = unicodedata.normalize("NFC", t)
    t = RE_INVISIBLE.sub("", t)
    t = RE_URL.sub(cfg["url_token"], t)
    t = RE_PUNCT_NO_SPACE.sub(r"\1 ", t)
    t = RE_DOT_NO_SPACE.sub(". ", t)
    n = int(cfg["max_repeated_chars"])
    t = re.sub(r"(\w)\1{%d,}" % n, lambda m: m.group(1) * n, t)   # "sooooo" -> "sooo"
    t = RE_REPEATED_PUNCT.sub(lambda m: m.group(1) * 3, t)       # "!!!!!!" -> "!!!"
    t = RE_SPACES.sub(" ", t).strip()
    return t


def normalize_for_dedup(text: str) -> str:
    """Clé de déduplication : insensible à la casse, à la ponctuation et aux espaces."""
    return RE_NON_WORD.sub(" ", str(text).lower()).strip()


def detect_language(text: str, ratio: float = 2.0) -> str:
    """Heuristique légère et explicable (pas de dépendance externe).

    - 'other'   : majorité de lettres non latines (grec, arabe…)
    - 'en', 'fr', 'es', 'it' : la langue dominante a au moins `ratio` fois plus
                  de mots-outils que la suivante
    - 'unknown' : pas assez d'indices (avis très court) -> on ne rejette pas
    """
    t = str(text)
    letters = RE_HAS_LETTER.findall(t)
    if letters and len(RE_LATIN_LETTER.findall(t)) / len(letters) < 0.5:
        return "other"
    words = RE_WORD.findall(t.lower())
    counts = sorted(((sum(w in fw for w in words), lang) for lang, fw in FUNCTION_WORDS.items()), reverse=True)
    (best_n, best), (second_n, _) = counts[0], counts[1]
    if best_n >= 1 and best_n > ratio * second_n:
        return best
    return "unknown"


def apply_cleaning(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Retourne (données propres, données rejetées, journal des règles)."""
    c = cfg["cleaning"]
    log: list[dict] = []
    df = df.copy()
    n0 = len(df)

    if "title_used" in df:
        log.append({"step": "titre ajouté devant l'avis (non redondant)", "rows_in": n0,
                    "rows_modified": int(df["title_used"].sum()), "rows_rejected": 0})

    df["text"] = df["text_raw"].map(lambda s: clean_text(s, c))
    df["text_modified"] = df["text"] != df["text_raw"].fillna("").str.strip()
    log.append({"step": "normalisation du texte", "rows_in": n0,
                "rows_modified": int(df["text_modified"].sum()), "rows_rejected": 0})

    df["reject_reason"] = pd.NA

    def reject(mask: pd.Series, reason: str, label: str) -> None:
        mask = mask & df["reject_reason"].isna()
        rows_in = int(df["reject_reason"].isna().sum())
        df.loc[mask, "reject_reason"] = reason
        log.append({"step": label, "rows_in": rows_in, "rows_modified": 0, "rows_rejected": int(mask.sum())})

    if "label_excluded" in df:
        reject(df["label_excluded"], "excluded_label", "note neutre écartée (règle de gestion)")
    reject(df["label"].isna(), "invalid_label", "note hors du référentiel")
    reject(df["text"].str.len() < int(c["min_chars"]), "empty_text", "texte vide ou inexploitable")
    reject(~df["text"].map(lambda s: bool(RE_HAS_LETTER.search(s))), "no_letters",
           "aucune lettre (ponctuation, chiffres ou émojis seuls)")

    df["language"] = df["text"].map(lambda s: detect_language(s, float(c["language_min_ratio"])))
    expected = c.get("expected_language")
    if expected:
        reject(~df["language"].isin([expected, "unknown"]), "wrong_language", f"avis dans une autre langue que '{expected}'")

    # Doublons : clé normalisée, calculée uniquement sur les lignes encore valides
    df["dedup_key"] = df["text"].map(normalize_for_dedup)
    alive = df["reject_reason"].isna()
    labels_per_key = df[alive].groupby("dedup_key")["label"].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    reject(df["dedup_key"].isin(conflicting), "duplicate_conflicting_labels",
           "doublons avec labels contradictoires (tous écartés)")
    alive = df["reject_reason"].isna()
    dup_mask = df[alive].duplicated("dedup_key", keep="first").reindex(df.index, fill_value=False)
    reject(dup_mask, "duplicate", "doublons (1ère occurrence conservée)")

    rejected = df[df["reject_reason"].notna()].copy()
    clean = df[df["reject_reason"].isna()].drop(columns=["reject_reason"]).copy()
    clean["label"] = clean["label"].astype(int)
    return clean.reset_index(drop=True), rejected.reset_index(drop=True), log


def save_cleaning_outputs(clean: pd.DataFrame, rejected: pd.DataFrame, cfg: dict) -> None:
    clean.to_parquet(ensure_parent(cfg["paths"]["processed"]), index=False)
    cols = ["review_id", "entity_id", "label_raw", "reject_reason", "language", "text_raw"]
    rejected[cols].to_csv(ensure_parent(cfg["paths"]["rejected"]), index=False, encoding="utf-8")
