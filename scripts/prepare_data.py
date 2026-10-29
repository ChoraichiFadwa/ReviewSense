"""Dépose le fichier source dans data/raw/ et produit un échantillon de démonstration.

Source : Trustpilot Reviews 123k (licence MIT)
https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k
À télécharger manuellement depuis l'onglet « Files and versions ».

    python scripts/prepare_data.py --file ~/Downloads/trustpilot_reviews.csv

Le fichier n'est pas modifié : il est copié tel quel (zone « raw » immuable),
avec son empreinte SHA-256 pour pouvoir prouver plus tard sur quelle version
des données le modèle a été entraîné.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k"
EXPECTED_COLUMNS = {"category", "company", "description", "title", "review", "stars"}
RAW_OUT = ROOT / "data" / "raw" / "trustpilot_reviews.csv"
SAMPLE_OUT = ROOT / "data" / "sample" / "trustpilot_sample.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="CSV Trustpilot téléchargé")
    p.add_argument("--sample-companies", type=int, default=160,
                   help="nb d'entreprises (entières) dans l'échantillon de démonstration")
    args = p.parse_args()

    src = Path(args.file).expanduser()
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes manquantes dans {src.name} : {sorted(missing)}")

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, RAW_OUT)
    digest = sha256(RAW_OUT)

    # Échantillon reproductible : des entreprises entières, pas des avis isolés
    companies = pd.Series(sorted(df["company"].unique())).sample(
        n=min(args.sample_companies, df["company"].nunique()), random_state=42)
    sample = df[df["company"].isin(set(companies))]
    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_OUT, index=False, encoding="utf-8")

    (RAW_OUT.parent / "SOURCE.md").write_text(
        "# Provenance des données\n\n"
        f"- Source : {SOURCE_URL}\n"
        "- Licence : MIT\n"
        f"- Fichier d'origine : `{src.name}`\n"
        f"- SHA-256 : `{digest}`\n"
        f"- Déposé le : {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n"
        f"- Lignes : {len(df)} ; entreprises : {df['company'].nunique()}\n",
        encoding="utf-8",
    )
    print(f"{len(df)} avis copiés dans {RAW_OUT.relative_to(ROOT)} (SHA-256 {digest[:12]}…)")
    print(f"{len(sample)} avis ({len(companies)} entreprises) dans {SAMPLE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
