"""Dataset references: URL safety and name extraction.

URL validation is a security boundary, so its tests are mostly refusals: everything that
could make the server fetch something it should not.
"""

from __future__ import annotations

import pytest

from app.services.dataset_analysis.references import (
    UnsafeURLError,
    extract_dataset_name,
    extract_urls,
    normalize_dataset_name,
    parse_dataset_url,
    validate_public_url,
)


class TestValidateUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.kaggle.com/datasets/heptapod/titanic",
            "https://kaggle.com/datasets/a/b",
            "https://huggingface.co/datasets/mnist",
            "https://archive.ics.uci.edu/dataset/53/iris",
            "https://www.openml.org/d/61",
        ],
    )
    def test_registry_urls_are_accepted(self, url):
        assert validate_public_url(url).startswith("https://")

    def test_http_is_upgraded_to_https(self):
        assert validate_public_url("http://www.kaggle.com/datasets/a/b").startswith("https://")

    def test_the_fragment_and_default_port_are_dropped(self):
        assert validate_public_url("https://www.kaggle.com:443/datasets/a/b#frag") == (
            "https://www.kaggle.com/datasets/a/b"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "https://user:pass@www.kaggle.com/datasets/a/b",  # credentials
            "https://www.kaggle.com@evil.example/datasets/a/b",  # userinfo trick
            "https://127.0.0.1/datasets/a/b",  # loopback literal
            "https://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
            "https://[::1]/datasets/a/b",  # IPv6 literal
            "https://localhost/datasets/a/b",
            "https://10.0.0.5/x",
            "https://www.kaggle.com:8443/datasets/a/b",  # non-default port
            "ftp://www.kaggle.com/datasets/a/b",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https://kaggle.com.evil.example/datasets/a/b",  # look-alike host
            "https://evil.example/?u=https://www.kaggle.com/datasets/a/b",
            "https://example.com/data.csv",  # not a researched registry
            "https://github.com/a/b",
            "https://www.kaggle.com/datasets/a b",  # whitespace
            "https://www.kaggle.com/datasets/a\nb",  # control character
            "",
            "   ",
            "not a url",
            "https://",
        ],
    )
    def test_hostile_or_unsupported_urls_are_refused(self, url):
        with pytest.raises(UnsafeURLError):
            validate_public_url(url)

    def test_an_overlong_url_is_refused(self):
        with pytest.raises(UnsafeURLError):
            validate_public_url("https://www.kaggle.com/datasets/a/" + "b" * 3000)

    def test_a_non_string_is_refused(self):
        with pytest.raises(UnsafeURLError):
            validate_public_url(None)  # type: ignore[arg-type]


class TestParseDatasetUrl:
    def test_kaggle(self):
        p = parse_dataset_url("https://www.kaggle.com/datasets/heptapod/titanic/data?select=train.csv")
        assert (p.source, p.identifier, p.name_hint) == ("kaggle", "heptapod/titanic", "titanic")

    def test_kaggle_competition_is_refused_with_a_reason(self):
        with pytest.raises(UnsafeURLError, match="competition"):
            parse_dataset_url("https://www.kaggle.com/competitions/titanic")

    def test_kaggle_non_dataset_page_is_refused(self):
        with pytest.raises(UnsafeURLError, match="not a dataset page"):
            parse_dataset_url("https://www.kaggle.com/someuser")

    def test_huggingface_owner_name(self):
        p = parse_dataset_url("https://huggingface.co/datasets/phihung/titanic/viewer/default/train")
        assert (p.source, p.identifier) == ("huggingface", "phihung/titanic")

    def test_huggingface_canonical_single_segment(self):
        assert parse_dataset_url("https://huggingface.co/datasets/mnist").identifier == "mnist"
        assert parse_dataset_url("https://huggingface.co/datasets/mnist/viewer").identifier == "mnist"

    def test_huggingface_model_page_is_refused(self):
        with pytest.raises(UnsafeURLError):
            parse_dataset_url("https://huggingface.co/bert-base-uncased")

    def test_uci_numeric_id(self):
        p = parse_dataset_url("https://archive.ics.uci.edu/dataset/53/iris")
        assert (p.source, p.identifier, p.name_hint) == ("uci", "53", "iris")

    def test_uci_legacy_name_url(self):
        p = parse_dataset_url("https://archive.ics.uci.edu/ml/datasets/Heart+Disease")
        assert (p.source, p.identifier) == ("uci", "Heart Disease")

    def test_openml_numeric_id(self):
        assert parse_dataset_url("https://www.openml.org/d/40945").identifier == "40945"

    def test_openml_search_url(self):
        p = parse_dataset_url("https://www.openml.org/search?type=data&sort=runs&id=61")
        assert (p.source, p.identifier) == ("openml", "61")

    def test_openml_other_pages_are_refused(self):
        with pytest.raises(UnsafeURLError):
            parse_dataset_url("https://www.openml.org/u/1")

    def test_the_canonical_url_is_https_without_a_fragment(self):
        p = parse_dataset_url("http://www.kaggle.com/datasets/a/b#x")
        assert p.canonical_url == "https://www.kaggle.com/datasets/a/b"


class TestExtractUrls:
    def test_trailing_punctuation_is_removed(self):
        assert extract_urls("see https://www.kaggle.com/datasets/a/b.") == [
            "https://www.kaggle.com/datasets/a/b"
        ]
        assert extract_urls("(https://huggingface.co/datasets/mnist), thanks") == [
            "https://huggingface.co/datasets/mnist"
        ]

    def test_duplicates_collapse_and_none_is_empty(self):
        text = "https://x.org/a https://x.org/a"
        assert extract_urls(text) == ["https://x.org/a"]
        assert extract_urls(None) == [] and extract_urls("") == []


class TestExtractDatasetName:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I have the Titanic dataset", "Titanic"),
            ("Analyze the Kaggle Titanic dataset.", "Titanic"),
            ("I downloaded the Heart Disease dataset", "Heart Disease"),
            ('my "Customer Churn 2024" dataset', "Customer Churn 2024"),
            ("a dataset called Mall Customers, please", "Mall Customers"),
            ("I uploaded student_grades.csv", "student grades"),
        ],
    )
    def test_the_named_dataset_is_extracted(self, text, expected):
        hint = extract_dataset_name(text)
        assert hint is not None and hint.name == expected

    def test_a_well_known_name_is_high_confidence(self):
        assert extract_dataset_name("I have the Iris dataset").confidence >= 0.9

    def test_a_bare_topic_word_is_flagged_generic_and_low_confidence(self):
        hint = extract_dataset_name("I downloaded a student dataset")
        assert hint.generic is True and hint.confidence < 0.5

    def test_a_bare_distinctive_name_is_found_without_the_word_dataset(self):
        assert extract_dataset_name("what can I build with MNIST?").name == "mnist"

    def test_a_topic_phrase_without_dataset_is_not_a_name(self):
        assert extract_dataset_name("predict diabetes") is None
        assert extract_dataset_name("I want to predict house prices") is None

    def test_discovery_verbs_are_not_names(self):
        assert extract_dataset_name("Recommend a dataset for plants") is None or (
            extract_dataset_name("Recommend a dataset for plants").name.lower() != "recommend"
        )

    def test_nothing_to_extract(self):
        assert extract_dataset_name(None) is None
        assert extract_dataset_name("hello there") is None


def test_dataset_name_normalisation_drops_the_generic_word():
    assert normalize_dataset_name("Titanic Dataset (CSV)") == "titanic"
    assert normalize_dataset_name("Heart-Disease  data set") == "heart disease"
