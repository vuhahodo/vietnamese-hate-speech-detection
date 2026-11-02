"""Feature extraction blocks for spam review classification.

Includes:
- B1: Word-level TF-IDF (1-2 gram) with pyvi word tokenization, min_df=2, sublinear_tf=True, max_features=50000.
- B2: Character-level TF-IDF (2-5 gram, analyzer=char), min_df=3, sublinear_tf=True, max_features=150000.
- B3: Surface statistical features (36 features) normalized by MaxAbsScaler fit on train.
- B4: Handcrafted spam dictionary based on spam type descriptions and Table 1 in Dinh et al. (arXiv:2207.14636, Section 3).
  Created strictly from domain description, never from the test set.

All blocks output scipy.sparse.csr_matrix, expose feature names, and support disk caching.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pyvi import ViTokenizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def _hash_strings(strings: Sequence[str]) -> str:
    """Compute deterministic SHA-256 hash for a sequence of strings."""
    hasher = hashlib.sha256()
    for s in strings:
        hasher.update(s.encode("utf-8", errors="replace"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


class BaseFeatureBlock(ABC):
    """Abstract base class for all feature extraction blocks."""

    def __init__(self, cache_dir: str | Path | None = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._is_fitted = False
        self._train_hash: str = ""
        self._feature_names: list[str] = []

    @property
    def names(self) -> list[str]:
        if not self._is_fitted:
            raise RuntimeError("Block has not been fitted yet.")
        return self._feature_names

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Return configuration dictionary for hashing."""
        pass

    def _get_cache_path(self, texts: Sequence[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:12]
        texts_hash = _hash_strings(texts)
        key = f"{self.__class__.__name__}_{config_hash}_{self._train_hash}_{texts_hash}"
        return self.cache_dir / f"{key}.npz"

    @abstractmethod
    def _fit_internal(self, train_texts: Sequence[str]) -> None:
        pass

    @abstractmethod
    def _transform_internal(self, texts: Sequence[str]) -> sparse.csr_matrix:
        pass

    def fit(self, train_texts: Sequence[str]) -> BaseFeatureBlock:
        self._train_hash = _hash_strings(train_texts)
        self._fit_internal(train_texts)
        self._is_fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        if not self._is_fitted:
            raise RuntimeError("Must fit before calling transform.")
        cache_path = self._get_cache_path(texts)
        if cache_path is not None and cache_path.exists():
            try:
                return sparse.load_npz(cache_path)
            except Exception:
                pass

        matrix = self._transform_internal(texts)
        if not isinstance(matrix, sparse.csr_matrix):
            matrix = matrix.tocsr()

        if cache_path is not None:
            try:
                sparse.save_npz(cache_path, matrix)
            except Exception:
                pass
        return matrix

    def fit_transform(self, train_texts: Sequence[str]) -> sparse.csr_matrix:
        self.fit(train_texts)
        return self.transform(train_texts)


class B1WordTfidf(BaseFeatureBlock):
    """B1: Word-level TF-IDF with pyvi tokenization, 1-2 gram, min_df=2, sublinear_tf=True, max_features=50000."""

    def __init__(
        self,
        min_df: int = 2,
        max_features: int = 50000,
        sublinear_tf: bool = True,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "name": "B1WordTfidf",
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
        }

    def _fit_internal(self, train_texts: Sequence[str]) -> None:
        tokenized = [ViTokenizer.tokenize(t) for t in train_texts]
        self.vectorizer.fit(tokenized)
        self._feature_names = [f"b1_{name}" for name in self.vectorizer.get_feature_names_out()]

    def _transform_internal(self, texts: Sequence[str]) -> sparse.csr_matrix:
        tokenized = [ViTokenizer.tokenize(t) for t in texts]
        return self.vectorizer.transform(tokenized)


class B2CharTfidf(BaseFeatureBlock):
    """B2: Character-level TF-IDF (2-5 gram, analyzer="char"), min_df=3, sublinear_tf=True, max_features=150000.

    Matches the VeriReview baseline (scripts/train_vispam_baseline.py), dev macro F1 0.8233 with balanced LR.
    """

    def __init__(
        self,
        min_df: int = 3,
        max_features: int = 150000,
        sublinear_tf: bool = True,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "name": "B2CharTfidf",
            "analyzer": "char",
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
        }

    def _fit_internal(self, train_texts: Sequence[str]) -> None:
        self.vectorizer.fit(train_texts)
        self._feature_names = [f"b2_{name}" for name in self.vectorizer.get_feature_names_out()]

    def _transform_internal(self, texts: Sequence[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(texts)


class B3Surface(BaseFeatureBlock):
    """B3: Surface statistical features (36 features) normalized by MaxAbsScaler fit on train."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:3|5|7|8|9)\d{8}\b|\b0\d{9,10}\b")
    VIETNAMESE_ACCENTS = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)
    EMOJI_PATTERN = re.compile(
        r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[:;=8][-o\*\']?[)\](\[dDpP/\:\}\{@\|\\]",
        re.UNICODE,
    )
    COMMON_ENGLISH_WORDS = {
        "good", "nice", "ok", "okay", "bad", "like", "love", "fast", "shipping", "delivery",
        "shop", "item", "quality", "product", "seller", "recommend", "thanks", "thank",
        "very", "best", "great", "cheap", "super", "perfect", "poor", "slow", "order",
    }

    FEATURE_NAMES = [
        "char_count", "word_count", "avg_word_length",
        "uppercase_count", "uppercase_ratio",
        "digit_count", "digit_ratio",
        "punctuation_count", "punctuation_ratio",
        "exclamation_count", "question_count", "period_count", "comma_count", "colon_count", "ellipsis_count",
        "repeated_char_max_len", "repeated_word_ratio", "whitespace_ratio",
        "has_url", "url_count",
        "has_phone", "phone_count",
        "emoji_count", "emoji_ratio",
        "non_vietnamese_latin_count", "non_vietnamese_latin_ratio",
        "accented_char_count", "accented_char_ratio",
        "unaccented_word_ratio",
        "english_word_count",
        "starts_with_capital",
        "all_caps_word_count",
        "price_currency_mention",
        "num_lines",
        "single_char_word_count",
        "max_token_len",
    ]

    def __init__(self, cache_dir: str | Path | None = DEFAULT_CACHE_DIR) -> None:
        super().__init__(cache_dir=cache_dir)
        self.scaler = MaxAbsScaler()
        self._feature_names = [f"b3_{name}" for name in self.FEATURE_NAMES]

    @property
    def config(self) -> dict[str, Any]:
        return {"name": "B3Surface", "num_features": len(self.FEATURE_NAMES)}

    @classmethod
    def _extract_row(cls, text: str) -> list[float]:
        c_count = len(text)
        words = text.split()
        w_count = len(words)
        avg_w_len = c_count / (w_count + 1e-5)

        upper_c = sum(1 for c in text if c.isupper())
        upper_r = upper_c / (c_count + 1e-5)

        digit_c = sum(1 for c in text if c.isdigit())
        digit_r = digit_c / (c_count + 1e-5)

        punct_c = sum(1 for c in text if c in string.punctuation)
        punct_r = punct_c / (c_count + 1e-5)

        excl_c = text.count("!")
        ques_c = text.count("?")
        period_c = text.count(".")
        comma_c = text.count(",")
        colon_c = text.count(":")
        ellipsis_c = len(re.findall(r"\.{2,}", text))

        # Repeated characters
        rep_matches = [len(m.group(0)) for m in re.finditer(r"(.)\1+", text)]
        rep_max_len = max(rep_matches) if rep_matches else 1

        # Repeated words
        rep_w_ratio = (w_count - len(set(words))) / (w_count + 1e-5)
        ws_ratio = sum(1 for c in text if c.isspace()) / (c_count + 1e-5)

        # URLs
        urls = cls.URL_PATTERN.findall(text)
        has_url = 1.0 if len(urls) > 0 else 0.0
        url_c = float(len(urls))

        # Phone numbers
        phones = cls.PHONE_PATTERN.findall(text)
        has_phone = 1.0 if len(phones) > 0 else 0.0
        phone_c = float(len(phones))

        # Emojis
        emojis = cls.EMOJI_PATTERN.findall(text)
        emoji_c = float(len(emojis))
        emoji_r = emoji_c / (c_count + 1e-5)

        # Accents & Latin
        accent_c = len(cls.VIETNAMESE_ACCENTS.findall(text))
        accent_r = accent_c / (c_count + 1e-5)

        # Non-vietnamese/latin characters
        latin_viet_c = sum(1 for c in text if c.isalnum() or c.isspace() or c in string.punctuation)
        non_viet_c = max(0, c_count - latin_viet_c)
        non_viet_r = non_viet_c / (c_count + 1e-5)

        # Unaccented words
        unaccented_w = sum(1 for w in words if not cls.VIETNAMESE_ACCENTS.search(w))
        unaccented_r = unaccented_w / (w_count + 1e-5)

        # English words
        eng_w = sum(1 for w in words if w.lower().strip(string.punctuation) in cls.COMMON_ENGLISH_WORDS)

        # Structure
        starts_cap = 1.0 if (c_count > 0 and text[0].isupper()) else 0.0
        all_caps_w = sum(1 for w in words if len(w) > 1 and w.isupper())
        currency_mentions = len(re.findall(r"\b(?:\d+k|\d+đ|vnd|đồng|nghìn|triệu)\b", text, re.I))
        num_lines = float(text.count("\n") + 1)
        single_char_w = sum(1 for w in words if len(w) == 1)
        max_tok_len = max([len(w) for w in words], default=0)

        return [
            float(c_count), float(w_count), float(avg_w_len),
            float(upper_c), float(upper_r),
            float(digit_c), float(digit_r),
            float(punct_c), float(punct_r),
            float(excl_c), float(ques_c), float(period_c), float(comma_c), float(colon_c), float(ellipsis_c),
            float(rep_max_len), float(rep_w_ratio), float(ws_ratio),
            has_url, url_c,
            has_phone, phone_c,
            emoji_c, emoji_r,
            float(non_viet_c), float(non_viet_r),
            float(accent_c), float(accent_r),
            float(unaccented_r),
            float(eng_w),
            starts_cap,
            float(all_caps_w),
            float(currency_mentions),
            num_lines,
            float(single_char_w),
            float(max_tok_len),
        ]

    def _fit_internal(self, train_texts: Sequence[str]) -> None:
        matrix = np.array([self._extract_row(t) for t in train_texts], dtype=np.float32)
        self.scaler.fit(matrix)

    def _transform_internal(self, texts: Sequence[str]) -> sparse.csr_matrix:
        matrix = np.array([self._extract_row(t) for t in texts], dtype=np.float32)
        scaled = self.scaler.transform(matrix)
        return sparse.csr_matrix(scaled)


class B4Dictionary(BaseFeatureBlock):
    """B4: Handcrafted dictionary of spam-related keywords.

    Handcrafted dictionary based on the spam type descriptions and Table 1
    in Dinh et al. (arXiv:2207.14636, Section 3). Created strictly from domain
    description, never from the test set.
    """

    GROUPS: dict[str, list[str]] = {
        "contact_social": [
            "zalo", "facebook", "fb", "fanpage", "inbox", "ib", "liên hệ", "hotline",
            "website", "link", "kết bạn", "tư vấn", "add zalo", "nhắn tin", "check ib",
            "sđt", "số điện thoại",
        ],
        "commercial_promo": [
            "giảm giá", "khuyến mãi", "khuyến mại", "khuyen mai", "sale", "voucher",
            "mua ngay", "đặt hàng", "freeship", "miễn phí ship", "ship cod", "tuyển ctv",
            "chiết khấu", "quà tặng", "ưu đãi", "giá rẻ", "chính hãng", "combo",
        ],
        "store_brand": [
            "shop", "cửa hàng", "gian hàng", "store", "shopee", "lazada", "tiki",
            "sendo", "tiktok shop", "đại lý", "chi nhánh", "thương hiệu",
        ],
        "generic_praise": [
            "hàng đẹp", "chất lượng tốt", "giao hàng nhanh", "đóng gói cẩn thận",
            "rất tốt", "tuyệt vời", "đáng tiền", "hài lòng", "ủng hộ shop", "chuẩn",
            "5 sao", "năm sao", "đẹp lắm", "ưng ý", "dùng tốt", "được", "ổn", "ok",
        ],
        "spam_coin_pattern": [
            "nhận xu", "lấy xu", "kiếm xu", "đánh giá nhận xu", "mang tính chất nhận xu",
            "hình ảnh nhận xu", "video nhận xu", "chưa dùng", "chưa thử", "mua giùm",
        ],
    }

    def __init__(self, cache_dir: str | Path | None = DEFAULT_CACHE_DIR) -> None:
        super().__init__(cache_dir=cache_dir)
        names = []
        for g_name, terms in self.GROUPS.items():
            names.append(f"b4_group_{g_name}_total")
            for t in terms:
                names.append(f"b4_term_{g_name}_{t.replace(' ', '_')}")
        self._feature_names = names

    @property
    def config(self) -> dict[str, Any]:
        return {"name": "B4Dictionary", "num_features": len(self._feature_names)}

    def _fit_internal(self, train_texts: Sequence[str]) -> None:
        # Dictionary features are rule-based; fitting records feature names and train hash
        pass

    def _transform_internal(self, texts: Sequence[str]) -> sparse.csr_matrix:
        rows = []
        for text in texts:
            lowered = text.casefold()
            row = []
            for g_name, terms in self.GROUPS.items():
                g_total = 0
                term_counts = []
                for t in terms:
                    cnt = lowered.count(t)
                    term_counts.append(float(cnt))
                    g_total += cnt
                row.append(float(g_total))
                row.extend(term_counts)
            rows.append(row)
        return sparse.csr_matrix(np.array(rows, dtype=np.float32))


class CompositeFeatureBlock(BaseFeatureBlock):
    """Combines multiple feature blocks horizontally via scipy.sparse.hstack."""

    def __init__(
        self,
        blocks: Sequence[BaseFeatureBlock],
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.blocks = list(blocks)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "name": "CompositeFeatureBlock",
            "sub_blocks": [b.config for b in self.blocks],
        }

    def _fit_internal(self, train_texts: Sequence[str]) -> None:
        names = []
        for b in self.blocks:
            b.fit(train_texts)
            names.extend(b.names)
        self._feature_names = names

    def _transform_internal(self, texts: Sequence[str]) -> sparse.csr_matrix:
        matrices = [b.transform(texts) for b in self.blocks]
        return sparse.hstack(matrices, format="csr")


def get_feature_block(
    spec: str,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
) -> BaseFeatureBlock:
    """Factory for feature blocks.

    Supports: 'b1', 'b2', 'b3', 'b4', and combinations joined by '+'
    e.g. 'b3+b4', 'b1+b2', 'b1+b2+b3+b4'.
    """
    key = spec.strip().lower()
    mapping: dict[str, type[BaseFeatureBlock]] = {
        "b1": B1WordTfidf,
        "b2": B2CharTfidf,
        "b3": B3Surface,
        "b4": B4Dictionary,
    }
    if "+" in key:
        parts = [p.strip() for p in key.split("+")]
        blocks = [mapping[p](cache_dir=cache_dir) for p in parts]
        return CompositeFeatureBlock(blocks=blocks, cache_dir=cache_dir)
    if key in mapping:
        return mapping[key](cache_dir=cache_dir)
    raise ValueError(f"Unknown feature block specification: {spec}")

