"""Profilage qualité : mêmes indicateurs mesurés AVANT et APRÈS nettoyage.

Le rapport produit (reports/data_quality_report.md) sert de preuve que le
nettoyage a fait ce qu'il annonce, et de base de discussion avec le métier
(ex. : "6 avis identiques ont des notes contradictoires : quelle est la
règle de gestion ?").
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .cleaning import (RE_DOT_NO_SPACE, RE_HAS_LETTER, RE_HTML_TAG, RE_PUNCT_NO_SPACE, RE_URL,
                       detect_language, normalize_for_dedup)
from .config import ensure_parent

RE_HTML_ENTITY = r"&[a-zA-Z]+;|&#\d+;"
RE_ELONGATION = r"(\w)\1{3,}"
RE_EMOJI = "[\U0001F300-\U0001FAFF☀-➿]"


def profile(df: pd.DataFrame, text_col: str, lang_ratio: float = 2.0, expected_language: str | None = None) -> dict:
    text = df[text_col].fillna("").astype(object).astype(str)
    length = text.str.len()
    key = text.map(normalize_for_dedup)
    is_dup = key.duplicated(keep=False)
    labels_by_key = df.assign(_k=key)[is_dup].groupby("_k")["label"].nunique()
    entities_by_key = df.assign(_k=key)[is_dup].groupby("_k")["entity_id"].nunique()
    lang = text.map(lambda s: detect_language(s, lang_ratio))
    no_letters = int(sum(not RE_HAS_LETTER.search(t) for t in text))
    per_entity = df.groupby("entity_id").size()

    def count(*patterns) -> int:
        # moteur `re` de Python (rétro-références supportées), indépendant du backend pandas
        rxs = [re.compile(p) if isinstance(p, str) else p for p in patterns]
        return int(sum(any(rx.search(t) for rx in rxs) for t in text))

    def dist(series: pd.Series) -> dict:
        vc = series.value_counts(dropna=False).sort_index()
        return {("exclu / invalide" if pd.isna(k) else str(k)): int(v) for k, v in vc.items()}

    out = {
        "lignes": int(len(df)),
        "entreprises distinctes": int(df["entity_id"].nunique()),
    }
    if "entity_category" in df:
        out["secteurs distincts"] = int(df["entity_category"].nunique())
    out.update({
        "avis par entreprise (médiane)": float(per_entity.median()),
        "avis par entreprise (max)": int(per_entity.max()),
        "répartition des notes source": dist(df["label_raw"].map(lambda v: str(v).strip())),
        "répartition des labels (1 / -1)": dist(df["label"].astype("Int64")),
        "textes vides": int((text.str.strip().str.len() == 0).sum()),
        "textes sans aucune lettre": no_letters,
        "longueur (car.) médiane": float(length.median()),
        "longueur (car.) p95": float(length.quantile(0.95)),
        "avis < 20 caractères": int((length < 20).sum()),
        "doublons (lignes en trop)": int(key.duplicated().sum()),
        "groupes de doublons": int(key[is_dup].nunique()),
        "groupes de doublons à labels contradictoires": int((labels_by_key > 1).sum()),
        "groupes de doublons sur plusieurs entreprises": int((entities_by_key > 1).sum()),
        "entités HTML (&amp; …)": count(RE_HTML_ENTITY),
        "balises HTML": count(RE_HTML_TAG),
        "URL": count(RE_URL),
        "ponctuation collée au mot suivant": count(RE_PUNCT_NO_SPACE, RE_DOT_NO_SPACE),
        "lettres répétées (sooooo)": count(RE_ELONGATION),
        "avis avec émojis": count(RE_EMOJI),
    })
    if expected_language:
        out[f"avis dans une autre langue que '{expected_language}'"] = int(
            (~lang.isin([expected_language, "unknown"])).sum())
    return out


def leakage_between_original_splits(df_raw: pd.DataFrame, split_col: str, text_col: str) -> int:
    """Nombre de textes présents dans plusieurs splits d'origine (si la source en fournit)."""
    key = df_raw[text_col].fillna("").map(normalize_for_dedup)
    return int(df_raw.assign(_k=key).groupby("_k")[split_col].nunique().gt(1).sum())


def write_report(before: dict, after: dict, cleaning_log: list[dict], split_summary: dict,
                 rejected_counts: dict, extra: dict, cfg: dict) -> None:
    reports = cfg["paths"]["reports_dir"]
    payload = {"avant_nettoyage": before, "après_nettoyage": after, "journal_nettoyage": cleaning_log,
               "rejets_par_motif": rejected_counts, "split": split_summary, "autres_constats": extra}
    with open(ensure_parent(reports / "data_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = ["# Rapport qualité des données", "",
             f"Source : `{cfg['source']['name']}` ({cfg['paths']['raw'].name})", "",
             "## Indicateurs avant / après nettoyage", "",
             "| Indicateur | Avant | Après |", "|---|---:|---:|"]
    for k in before:
        b, a = before[k], after.get(k, "")
        if isinstance(b, dict) and not isinstance(a, dict):
            a = {}
        if isinstance(b, dict):
            b = ", ".join(f"{kk}: {vv:,}".replace(",", " ") for kk, vv in b.items())
            a = ", ".join(f"{kk}: {vv:,}".replace(",", " ") for kk, vv in a.items())
        lines.append(f"| {k} | {_fmt(b)} | {_fmt(a)} |")

    lines += ["", "## Journal des règles de nettoyage", "",
              "| Étape | Lignes en entrée | Lignes modifiées | Lignes rejetées |", "|---|---:|---:|---:|"]
    for s in cleaning_log:
        lines.append(f"| {s['step']} | {_fmt(s['rows_in'])} | {_fmt(s['rows_modified'])} | {_fmt(s['rows_rejected'])} |")

    lines += ["", "## Lignes en quarantaine (data/rejected/)", "", "| Motif | Lignes |", "|---|---:|"]
    lines += [f"| `{k}` | {_fmt(v)} |" for k, v in rejected_counts.items()]

    lines += ["", "## Découpage train / validation / test", "",
              "| Split | Avis | Entités | % positifs |", "|---|---:|---:|---:|"]
    for name, s in split_summary["splits"].items():
        lines.append(f"| {name} | {_fmt(s['avis'])} | {_fmt(s['entités'])} | {s['pct_positifs']:.1f} % |")
    lines += ["", f"- Entités présentes dans plusieurs splits : **{split_summary['entités_partagées']}**",
              f"- Textes présents dans plusieurs splits : **{split_summary['textes_partagés']}**"]

    if extra:
        lines += ["", "## Autres constats", ""]
        lines += [f"- {k} : **{_fmt(v)}**" for k, v in extra.items()]

    with open(ensure_parent(reports / "data_quality_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    if isinstance(v, float):
        return f"{v:,.0f}".replace(",", " ") if v.is_integer() else f"{v:.2f}"
    return str(v)
