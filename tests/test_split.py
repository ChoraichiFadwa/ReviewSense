"""Tests anti-fuite du découpage train / validation / test."""
import pandas as pd
import pytest

from src.split import check_no_leakage, split_dataset


def _data(n_entities=60, per_entity=10):
    rows = [{"review_id": f"{e}-{i}", "entity_id": f"c{e}", "text": f"review {e} {i}",
             "dedup_key": f"review {e} {i}", "label": 1 if i % 2 else -1}
            for e in range(n_entities) for i in range(per_entity)]
    return pd.DataFrame(rows)


def test_no_entity_and_no_text_shared_between_splits():
    cfg = {"seed": 0, "split": {"test_size": 0.1, "val_size": 0.1, "group_by_entity": True}}
    parts = split_dataset(_data(), cfg)
    assert sum(len(p) for p in parts.values()) == 600
    ents = [set(p["entity_id"]) for p in parts.values()]
    assert not (ents[0] & ents[1]) and not (ents[0] & ents[2]) and not (ents[1] & ents[2])


def test_leakage_check_blocks_a_bad_split():
    df = _data(4, 2)
    bad = {"train": df.iloc[:5], "val": df.iloc[4:6], "test": df.iloc[6:]}   # ligne 4 dans train ET val
    with pytest.raises(AssertionError, match="Fuite"):
        check_no_leakage(bad)
