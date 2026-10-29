"""Découpage train / validation / test SANS fuite de données.

- Le dédoublonnage est fait AVANT le découpage : un même texte ne peut pas
  être à la fois dans le train et dans le test.
- Découpage groupé par entreprise : le modèle
  est évalué sur des entités qu'il n'a jamais vues, ce qui correspond à l'usage
  réel (analyser la réputation d'une nouvelle entreprise). Sinon il pourrait
  apprendre le nom ou les produits d'une entreprise et paraître meilleur qu'il n'est.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import ensure_parent

SPLITS = ("train", "val", "test")


def split_dataset(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    s, seed = cfg["split"], cfg["seed"]
    test_size, val_size = float(s["test_size"]), float(s["val_size"])

    if s.get("group_by_entity", True):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trval_idx, test_idx = next(gss.split(df, groups=df["entity_id"]))
        trval, test = df.iloc[trval_idx], df.iloc[test_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
        tr_idx, val_idx = next(gss.split(trval, groups=trval["entity_id"]))
        train, val = trval.iloc[tr_idx], trval.iloc[val_idx]
    else:
        trval, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
        train, val = train_test_split(trval, test_size=val_size / (1 - test_size),
                                      stratify=trval["label"], random_state=seed)

    parts = {"train": train, "val": val, "test": test}
    check_no_leakage(parts, group_by_entity=s.get("group_by_entity", True))
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def check_no_leakage(parts: dict[str, pd.DataFrame], group_by_entity: bool = True) -> None:
    """Contrôle bloquant : le pipeline s'arrête si une fuite est détectée."""
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_text = set(parts[a]["dedup_key"]) & set(parts[b]["dedup_key"])
            assert not shared_text, f"Fuite : {len(shared_text)} textes communs entre {a} et {b}"
            if group_by_entity:
                shared_ent = set(parts[a]["entity_id"]) & set(parts[b]["entity_id"])
                assert not shared_ent, f"Fuite : {len(shared_ent)} entités communes entre {a} et {b}"


def summarize(parts: dict[str, pd.DataFrame]) -> dict:
    ents = [set(p["entity_id"]) for p in parts.values()]
    keys = [set(p["dedup_key"]) for p in parts.values()]
    shared_e = sum(len(ents[i] & ents[j]) for i in range(3) for j in range(i + 1, 3))
    shared_t = sum(len(keys[i] & keys[j]) for i in range(3) for j in range(i + 1, 3))
    return {
        "splits": {k: {"avis": int(len(v)), "entités": int(v["entity_id"].nunique()),
                       "pct_positifs": float((v["label"] == 1).mean() * 100)} for k, v in parts.items()},
        "entités_partagées": shared_e,
        "textes_partagés": shared_t,
    }


def save_splits(parts: dict[str, pd.DataFrame], cfg: dict) -> None:
    cols = ["review_id", "entity_id", "text", "label", "label_raw"]
    for name, part in parts.items():
        part[cols].to_parquet(ensure_parent(cfg["paths"]["splits_dir"] / f"{name}.parquet"), index=False)


def load_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(cfg["paths"]["splits_dir"] / f"{name}.parquet") for name in SPLITS}
