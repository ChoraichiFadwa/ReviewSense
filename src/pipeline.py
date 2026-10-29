"""Lance toute la chaîne : données -> modèle -> réputation -> datamart.

    python -m src.pipeline                              # jeu complet (~4 min)
    python -m src.pipeline --config config.sample.yaml  # échantillon (< 1 min)
"""
from __future__ import annotations

import argparse
import logging
import time

from . import data_pipeline, datamart, reputation, train
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-data", action="store_true", help="réutiliser les splits déjà produits")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-10s | %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)

    t0 = time.perf_counter()
    if not args.skip_data:
        data_pipeline.run(cfg)
    train.run(cfg)
    reputation.run(cfg)
    datamart.run(cfg)
    logging.getLogger("pipeline").info("Terminé en %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
