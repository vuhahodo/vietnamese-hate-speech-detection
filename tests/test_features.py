"""Tests for vispam.features module."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from vispam.features import (
    B1WordTfidf,
    B2CharTfidf,
    B3Surface,
    B4Dictionary,
    CompositeFeatureBlock,
    get_feature_block,
)

SAMPLE_TRAIN = [
    "Sản phẩm rất tốt, giao hàng nhanh chóng!",
    "Shop lừa đảo, liên hệ qua zalo 0901234567 để biết thêm chi tiết.",
    "Hàng dùng tạm được, đóng gói cẩn thận 5 sao nhé.",
    "Mua ngay hôm nay giảm giá 50% freeship toàn quốc https://shop.vn",
]

SAMPLE_TEST = [
    "Sản phẩm ổn, đáng tiền.",
    "Zalo 0912345678 shop ơi.",
]


@pytest.mark.parametrize("block_cls", [B1WordTfidf, B2CharTfidf, B3Surface, B4Dictionary])
def test_individual_blocks(block_cls, tmp_path):
    block = block_cls(cache_dir=tmp_path)
    X_train = block.fit_transform(SAMPLE_TRAIN)

    assert isinstance(X_train, sparse.csr_matrix)
    assert X_train.shape[0] == len(SAMPLE_TRAIN)
    assert X_train.shape[1] == len(block.names)
    assert X_train.shape[1] > 0

    # Transform test set
    X_test = block.transform(SAMPLE_TEST)
    assert isinstance(X_test, sparse.csr_matrix)
    assert X_test.shape[0] == len(SAMPLE_TEST)
    assert X_test.shape[1] == X_train.shape[1]


def test_caching(tmp_path):
    block = B3Surface(cache_dir=tmp_path)
    X1 = block.fit_transform(SAMPLE_TRAIN)

    # Check cache file exists in tmp_path
    cache_files = list(tmp_path.glob("*.npz"))
    assert len(cache_files) == 1

    # Second transform call should load from cache
    X2 = block.transform(SAMPLE_TRAIN)
    assert np.allclose(X1.toarray(), X2.toarray())


def test_composite_block(tmp_path):
    block = get_feature_block("b3+b4", cache_dir=tmp_path)
    X = block.fit_transform(SAMPLE_TRAIN)

    b3 = B3Surface(cache_dir=tmp_path).fit(SAMPLE_TRAIN)
    b4 = B4Dictionary(cache_dir=tmp_path).fit(SAMPLE_TRAIN)

    expected_cols = len(b3.names) + len(b4.names)
    assert X.shape[1] == expected_cols
    assert len(block.names) == expected_cols


def test_b4_dictionary_detection(tmp_path):
    block = B4Dictionary(cache_dir=tmp_path)
    X = block.fit_transform(["liên hệ qua zalo hoặc facebook"])
    # "zalo" and "facebook" should have positive count
    names = block.names
    zalo_idx = names.index("b4_term_contact_social_zalo")
    fb_idx = names.index("b4_term_contact_social_facebook")
    arr = X.toarray()[0]
    assert arr[zalo_idx] == 1.0
    assert arr[fb_idx] == 1.0

