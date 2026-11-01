import pytest

from vispam.data import DEFAULT_PATH, load_splits, normalize

needs_data = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="ViSpamReviews CSV not downloaded")


def test_normalize_collapses_whitespace_and_nfc():
    assert normalize("  á  b\n") == "á b"


@needs_data
def test_split_sizes_match_ai4life_audit():
    splits, audit = load_splits()
    assert audit["source_splits"] == {"train": 14306, "dev": 1590, "test": 3974}
    assert audit["used_splits"] == {"train": 14181, "dev": 1587, "test": 3971}


@needs_data
def test_no_text_shared_across_splits():
    splits, _ = load_splits()
    keys = {part: set(df.text.str.casefold()) for part, df in splits.items()}
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["dev"]
    assert not keys["dev"] & keys["test"]


@needs_data
def test_task_labels_consistent():
    splits, _ = load_splits()
    for df in splits.values():
        assert ((df.label == 0) == (df.spam_type == 0)).all()
