"""Tests des règles de nettoyage : chaque règle annoncée dans le README est vérifiée ici."""
import pandas as pd

from src.cleaning import apply_cleaning, clean_text, detect_language, normalize_for_dedup

CFG = {"min_chars": 2, "max_repeated_chars": 3, "url_token": " linkurl ",
       "expected_language": "en", "language_min_ratio": 2.0}


def test_html_is_unescaped_and_tags_removed():
    assert clean_text("Great &amp; fast<br/>delivery", CFG) == "Great & fast delivery"


def test_urls_are_replaced_by_a_token():
    assert "linkurl" in clean_text("See https://example.com/x?y=1 for details", CFG)
    assert "example.com" not in clean_text("See https://example.com/x?y=1 for details", CFG)


def test_space_added_after_glued_punctuation():
    assert clean_text("Good service,very fast.Will order again", CFG) == "Good service, very fast. Will order again"


def test_decimals_are_not_split():
    assert clean_text("Paid 4.99 for delivery", CFG) == "Paid 4.99 for delivery"


def test_elongations_are_reduced():
    assert clean_text("Sooooo goooood!!!!!!", CFG) == "Sooo goood!!!"


def test_negations_are_kept():
    # la négation porte la polarité : elle ne doit jamais disparaître
    assert clean_text("Not good, never again", CFG) == "Not good, never again"


def test_cleaning_is_idempotent():
    once = clean_text("Bad<br>service!!!!!!   avoid.They lie", CFG)
    assert clean_text(once, CFG) == once


def test_dedup_key_ignores_case_and_punctuation():
    assert normalize_for_dedup("Great service!!") == normalize_for_dedup("great   SERVICE")


def test_language_detection():
    assert detect_language("Delivered on time, exactly as described") == "en"
    assert detect_language("Très contente de la collection, je recommande") == "fr"
    assert detect_language("Great service") == "unknown"     # trop court pour décider : on garde
    assert detect_language("γνωρησα ανθρωπους που ηταν") == "other"


def _frame(rows):
    df = pd.DataFrame(rows, columns=["review_id", "entity_id", "text_raw", "label_raw", "label", "label_excluded"])
    return df


def test_rejections_are_traced_with_a_reason():
    df = _frame([
        ("a", "c1", "Great service, fast delivery", "5", 1.0, False),
        ("b", "c1", "great service fast delivery!!", "4", 1.0, False),     # doublon -> 1 seul gardé
        ("c", "c2", "It was ok I guess", "3", None, True),                  # note neutre exclue
        ("d", "c2", "........", "1", -1.0, False),                          # aucune lettre
        ("e", "c3", "Très contente de la collection, je recommande", "5", 1.0, False),  # langue
        ("f", "c3", "Never again, the worst", "1", -1.0, False),
    ])
    clean, rejected, log = apply_cleaning(df, {"cleaning": CFG})
    assert set(clean["review_id"]) == {"a", "f"}
    reasons = dict(zip(rejected["review_id"], rejected["reject_reason"]))
    assert reasons == {"b": "duplicate", "c": "excluded_label", "d": "no_letters", "e": "wrong_language"}
    assert sum(s["rows_rejected"] for s in log) == len(rejected)


def test_duplicates_with_conflicting_labels_are_all_removed():
    df = _frame([
        ("a", "c1", "Fine", "5", 1.0, False),
        ("b", "c2", "fine.", "1", -1.0, False),
        ("c", "c2", "Terrible support", "1", -1.0, False),
    ])
    clean, rejected, _ = apply_cleaning(df, {"cleaning": CFG})
    assert list(clean["review_id"]) == ["c"]
    assert set(rejected["reject_reason"]) == {"duplicate_conflicting_labels"}
