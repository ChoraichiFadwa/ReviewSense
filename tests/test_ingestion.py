"""Tests du contrat d'ingestion : mapping des colonnes et des notes."""
import pandas as pd
import pytest

from src.ingestion import SchemaError, combine_title_review, map_labels, standardize

SOURCE = {
    "name": "test",
    "columns": {"text": "review", "title": "title", "label": "stars", "entity": "company"},
    "entity_attributes": {"category": "category"},
    "entity_pattern": None,
    "label_mapping": {1: -1, 2: -1, 4: 1, 5: 1},
    "excluded_labels": [3],
}


def test_star_mapping_handles_text_and_float_exports():
    raw = pd.Series(["1", "2.0", "4", " 5 ", "3", "abc"])
    out = map_labels(raw, SOURCE["label_mapping"])
    assert out.tolist()[:4] == [-1, -1, 1, 1]
    assert out.iloc[4:].isna().all()   # 3 = exclue, "abc" = invalide -> rejetées plus loin, pas devinées


def test_missing_column_fails_early_with_a_clear_message():
    df = pd.DataFrame({"review": ["x"], "stars": ["5"], "company": ["c"]})   # ni title ni category
    with pytest.raises(SchemaError, match="Colonnes absentes"):
        standardize(df, SOURCE, "fichier.csv")


def test_title_is_not_added_when_it_is_the_truncated_start_of_the_review():
    title = pd.Series(["Great quality dog robe although…", "Life saver", ""])
    review = pd.Series(["Great quality dog robe although it took a while", "Bought it in May", "Fine"])
    text, used = combine_title_review(title, review)
    assert text.tolist() == ["Great quality dog robe although it took a while",
                             "Life saver. Bought it in May", "Fine"]
    assert used.tolist() == [False, True, False]


def test_standardize_produces_the_standard_schema():
    df = pd.DataFrame({"review": ["Bad", "Ok"], "title": ["Avoid", "Meh"], "stars": ["1", "3"],
                       "company": [" acme.com ", "acme.com"], "category": ["Shops", "Shops"]})
    out = standardize(df, SOURCE, "fichier.csv")
    assert {"review_id", "entity_id", "entity_category", "text_raw", "label", "label_excluded"} <= set(out.columns)
    assert out["entity_id"].tolist() == ["acme.com", "acme.com"]          # espaces parasites retirés
    assert out["label_excluded"].tolist() == [False, True]
    assert out["review_id"].is_unique
