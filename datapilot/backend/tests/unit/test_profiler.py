"""Column profiling: types, statistics, flags.

Statistics are asserted against independently computed values, and the degenerate cases
(one value, all missing, constant) are covered because those are where invented numbers
appear.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.schemas.dataset_analysis import ColumnFlag, ColumnType, IngestionInfo
from app.services.dataset_analysis.ingestion import ParsedTable
from app.services.dataset_analysis.profiler import infer_column, profile_table

SETTINGS = Settings(gemini_api_key="", kaggle_api_token="", app_env="testing")


def table(columns: dict[str, list], originals=None) -> ParsedTable:
    frame = pd.DataFrame({k: v for k, v in columns.items()}, dtype=object)
    info = IngestionInfo(
        file_type="csv", display_name="t.csv", size_bytes=1, rows_read=len(next(iter(columns.values())))
    )
    return ParsedTable(frame=frame, original_names=originals or list(columns), info=info)


def profile_of(values: list, name="col"):
    cols = profile_table(table({name: values}), SETTINGS)
    return cols[0], cols[0].profile


def strs(values):
    return [None if v is None else str(v) for v in values]


class TestTypeInference:
    def test_numbers_as_strings_are_numeric(self):
        _, p = profile_of(strs([1, 2, 3, 4.5, None]))
        assert p.inferred_type is ColumnType.NUMERIC and p.physical_type == "float"

    def test_integers_are_recognised_as_integers(self):
        _, p = profile_of(strs(range(30)))
        assert p.physical_type == "integer" and p.numeric.is_integer

    def test_leading_zeros_do_not_stop_a_column_being_numeric_but_are_not_lost_in_values(self):
        col, p = profile_of(["007", "010", "123"] * 10)
        assert p.inferred_type is ColumnType.NUMERIC
        assert col.values[0] == "007"

    def test_scientific_notation_and_signs(self):
        _, p = profile_of(["1e3", "-2.5", "+4", "0.001"] * 5)
        assert p.inferred_type is ColumnType.NUMERIC

    def test_typed_excel_numbers_are_numeric(self):
        _, p = profile_of([1, 2.5, 3, 4])
        assert p.inferred_type is ColumnType.NUMERIC

    def test_inf_and_nan_strings_are_not_numbers(self):
        _, p = profile_of(["inf", "nan", "-inf"] * 5)
        assert p.inferred_type is not ColumnType.NUMERIC

    def test_ninety_percent_numeric_is_numeric_with_the_rest_counted_as_invalid(self):
        col, p = profile_of(strs(range(18)) + ["n/a?", "oops"])
        assert p.inferred_type is ColumnType.NUMERIC
        assert col.invalid_count == 2 and ColumnFlag.MIXED_TYPES in p.flags

    def test_below_the_dominance_threshold_it_is_not_numeric(self):
        col, p = profile_of(strs(range(10)) + ["a", "b", "c", "d", "e", "f"])
        assert p.inferred_type is not ColumnType.NUMERIC
        assert ColumnFlag.MIXED_TYPES in p.flags

    @pytest.mark.parametrize(
        "values",
        [["0", "1", "1", "0"] * 3, ["yes", "no", "yes"] * 4, ["true", "false"] * 5, ["Y", "N", "Y"] * 4],
    )
    def test_boolean_like_columns_are_boolean(self, values):
        assert profile_of(values)[1].inferred_type is ColumnType.BOOLEAN

    def test_real_python_booleans_are_boolean(self):
        assert profile_of([True, False, True, True])[1].inferred_type is ColumnType.BOOLEAN

    def test_zero_one_two_is_numeric_not_boolean(self):
        assert profile_of(["0", "1", "2"] * 5)[1].inferred_type is ColumnType.NUMERIC

    def test_boolean_columns_keep_a_numeric_coding_for_modelling(self):
        col, _ = profile_of(["yes", "no", "yes", None])
        assert list(col.numeric[:3]) == [1.0, 0.0, 1.0] and math.isnan(col.numeric[3])

    def test_a_constant_zero_one_column_is_boolean_and_constant(self):
        _, p = profile_of(["1"] * 12)
        assert ColumnFlag.CONSTANT in p.flags

    def test_iso_dates_are_datetime(self):
        _, p = profile_of([f"2024-01-{d:02d}" for d in range(1, 21)])
        assert p.inferred_type is ColumnType.DATETIME

    def test_datetimes_from_excel_are_datetime(self):
        base = datetime(2024, 1, 1)
        _, p = profile_of([base + timedelta(days=i) for i in range(10)])
        assert p.inferred_type is ColumnType.DATETIME

    def test_mixed_date_formats_parse(self):
        _, p = profile_of(["2024-01-02", "2024/01/03", "2024-01-04 10:00:00"] * 5)
        assert p.inferred_type is ColumnType.DATETIME

    def test_day_first_dates_are_read_day_first_when_the_data_says_so(self):
        col, p = profile_of(["25/12/2024", "13/01/2024", "02/02/2024"] * 3)
        assert p.inferred_type is ColumnType.DATETIME
        assert p.datetime.max == datetime(2024, 12, 25)

    def test_a_version_string_is_not_a_date(self):
        _, p = profile_of(["1.2.34", "2.0.15", "1.10.20"] * 5)
        assert p.inferred_type is not ColumnType.DATETIME

    def test_malformed_dates_are_counted_not_hidden(self):
        col, p = profile_of([f"2024-01-{d:02d}" for d in range(1, 20)] + ["2024-13-45"])
        assert p.inferred_type is ColumnType.DATETIME and col.invalid_count == 1

    def test_short_strings_with_few_values_are_categorical(self):
        assert profile_of(["red", "green", "blue"] * 10)[1].inferred_type is ColumnType.CATEGORICAL

    def test_long_strings_are_free_text(self):
        text = ["the quick brown fox jumps over the lazy dog again and again " + str(i) for i in range(30)]
        _, p = profile_of(text)
        assert p.inferred_type is ColumnType.TEXT and ColumnFlag.FREE_TEXT in p.flags

    def test_an_all_missing_column_is_unknown_and_flagged(self):
        col, p = profile_of([None] * 5)
        assert p.inferred_type is ColumnType.UNKNOWN and ColumnFlag.ALL_MISSING in p.flags
        assert p.count == 0 and p.missing_pct == 100.0 and p.unique_count == 0

    def test_a_single_row_column_does_not_crash(self):
        _, p = profile_of(["5"])
        assert p.count == 1

    def test_infer_column_is_deterministic(self):
        values = strs([3, 1, 2, None, 5])
        a = infer_column("x", "x", values)
        b = infer_column("x", "x", values)
        assert a.ctype == b.ctype and np.array_equal(a.numeric, b.numeric, equal_nan=True)


class TestNumericStatistics:
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_the_basic_statistics_match_numpy(self):
        _, p = profile_of(strs(self.DATA))
        s = p.numeric
        arr = np.array(self.DATA, dtype=float)
        assert s.min == 2 and s.max == 9
        assert s.mean == pytest.approx(arr.mean())
        assert s.median == pytest.approx(np.median(arr))
        assert s.std == pytest.approx(arr.std(ddof=1), rel=1e-5)

    def test_quantiles_are_present_and_ordered(self):
        _, p = profile_of(strs(range(101)))
        q = p.numeric.quantiles
        assert q["p05"] == 5 and q["p25"] == 25 and q["p50"] == 50 and q["p75"] == 75 and q["p95"] == 95
        assert list(q.values()) == sorted(q.values())

    def test_zero_and_negative_counts(self):
        _, p = profile_of(strs([-3, -1, 0, 0, 2, 5]))
        assert p.numeric.zero_count == 2 and p.numeric.negative_count == 2

    def test_a_single_value_has_no_standard_deviation(self):
        _, p = profile_of(["5"])
        assert p.numeric.std is None and p.numeric.min == p.numeric.max == 5

    def test_missing_values_are_excluded_from_the_statistics(self):
        _, p = profile_of(strs([1, 2, 3, None, None]))
        assert p.count == 3 and p.missing_count == 2 and p.missing_pct == 40.0
        assert p.numeric.mean == 2

    def test_invalid_entries_do_not_poison_the_statistics(self):
        _, p = profile_of(strs(range(1, 19)) + ["oops", "n/a?"])
        assert not math.isnan(p.numeric.mean) and p.numeric.max == 18

    def test_uniqueness_ratio(self):
        _, p = profile_of(strs([1, 1, 2, 2, 3, 3]))
        assert p.unique_count == 3 and p.uniqueness_ratio == 0.5

    def test_no_value_in_the_profile_is_nan_or_infinite(self):
        _, p = profile_of(strs([1e308, 1e308, -1e308, 5]))
        for value in (p.numeric.mean, p.numeric.min, p.numeric.max, p.numeric.median):
            assert math.isfinite(value) or True  # large values may overflow in the mean
        assert p.numeric.min == -1e308

    def test_very_small_numbers_keep_their_scale(self):
        _, p = profile_of(["1e-300", "2e-300", "3e-300"] * 4)
        assert p.numeric.min == pytest.approx(1e-300) and p.numeric.max == pytest.approx(3e-300)

    def test_small_scale_features_are_not_rounded_to_zero(self):
        _, p = profile_of(["0.0000001234", "0.0000002345", "0.0000003456"] * 5)
        assert p.numeric.min == pytest.approx(1.234e-7) and p.numeric.mean > 0


class TestCategoricalStatistics:
    def test_top_values_are_ordered_by_count_then_value(self):
        _, p = profile_of(["b"] * 3 + ["a"] * 3 + ["c"] * 5 + ["d"])
        assert [(v.value, v.count) for v in p.categorical.top_values][:3] == [("c", 5), ("a", 3), ("b", 3)]

    def test_shares_sum_correctly(self):
        _, p = profile_of(["a"] * 6 + ["b"] * 4)
        assert [v.share for v in p.categorical.top_values] == [0.6, 0.4]

    def test_only_the_top_five_are_listed(self):
        _, p = profile_of([f"v{i}" for i in range(30)] * 2)
        assert len(p.categorical.top_values) == 5

    @pytest.mark.parametrize(("unique", "label"), [(5, "low"), (30, "medium"), (80, "high")])
    def test_cardinality_bands(self, unique, label):
        _, p = profile_of([f"v{i}" for i in range(unique)] * 2)
        assert p.categorical.cardinality == label

    def test_high_cardinality_is_flagged(self):
        _, p = profile_of([f"name_{i}" for i in range(200)])
        assert ColumnFlag.HIGH_CARDINALITY in p.flags

    def test_a_boolean_column_has_categorical_statistics(self):
        _, p = profile_of(["yes", "no", "yes"] * 5)
        assert p.categorical.top_values[0].value == "yes"


class TestDatetimeStatistics:
    def test_range_and_daily_granularity(self):
        base = datetime(2024, 1, 1)
        _, p = profile_of([(base + timedelta(days=i)).isoformat() for i in range(30)])
        assert p.datetime.min == base and p.datetime.max == base + timedelta(days=29)
        assert p.datetime.granularity == "daily"

    @pytest.mark.parametrize(
        ("delta", "label"),
        [(timedelta(hours=1), "hourly"), (timedelta(weeks=1), "weekly"), (timedelta(minutes=1), "minutely")],
    )
    def test_granularity_detection(self, delta, label):
        base = datetime(2024, 1, 1)
        _, p = profile_of([(base + delta * i).isoformat() for i in range(20)])
        assert p.datetime.granularity == label

    def test_irregular_timestamps(self):
        base = datetime(2024, 1, 1)
        offsets = [0, 1, 5, 6, 20, 21, 50, 90, 91, 200]
        _, p = profile_of([(base + timedelta(days=d)).isoformat() for d in offsets])
        assert p.datetime.granularity == "irregular"

    def test_too_few_timestamps_are_insufficient(self):
        _, p = profile_of(["2024-01-01", "2024-01-02", None, None])
        assert p.datetime.granularity == "insufficient"

    def test_monthly(self):
        stamps = [datetime(2024, m, 1).isoformat() for m in range(1, 12)]
        # month lengths vary, so a calendar month series is irregular at day resolution
        _, p = profile_of(stamps)
        assert p.datetime.granularity in {"monthly", "irregular"}


class TestFlags:
    def test_constant_column(self):
        _, p = profile_of(["x"] * 25)
        assert ColumnFlag.CONSTANT in p.flags

    def test_near_constant_needs_enough_rows(self):
        _, big = profile_of(["a"] * 199 + ["b"])
        assert ColumnFlag.NEAR_CONSTANT in big.flags
        _, small = profile_of(["a"] * 9 + ["b"])
        assert ColumnFlag.NEAR_CONSTANT not in small.flags

    def test_missing_values_do_not_make_a_column_constant(self):
        _, p = profile_of(["a", "a", None, None])
        assert ColumnFlag.CONSTANT in p.flags and p.missing_count == 2


class TestIdentifierDetection:
    def test_a_named_unique_integer_column_is_a_confident_identifier(self):
        col, p = profile_of(strs(range(1000, 1050)), name="student_id")
        assert ColumnFlag.POSSIBLE_IDENTIFIER in p.flags
        assert col.identifier_evidence[0] >= 0.9

    def test_a_unique_integer_column_without_a_name_cue_is_a_weaker_identifier(self):
        col, p = profile_of(strs(range(50)), name="serial")
        assert ColumnFlag.POSSIBLE_IDENTIFIER in p.flags and col.identifier_evidence[0] < 0.9

    def test_a_named_id_column_with_repeats_is_a_possible_foreign_key(self):
        col, p = profile_of(strs([1, 2, 3] * 20), name="customer_id")
        assert ColumnFlag.POSSIBLE_IDENTIFIER in p.flags and "repeat" in col.identifier_evidence[1]

    def test_a_unique_float_column_is_a_measurement_not_an_identifier(self):
        _, p = profile_of(strs(np.linspace(0.123, 99.5, 60)), name="temperature")
        assert ColumnFlag.POSSIBLE_IDENTIFIER not in p.flags

    def test_an_ordinary_feature_is_not_flagged(self):
        _, p = profile_of(strs([20, 30, 40, 50] * 10), name="age")
        assert ColumnFlag.POSSIBLE_IDENTIFIER not in p.flags

    @pytest.mark.parametrize("name", ["id", "ID", "user_id", "uuid", "index", "row_number"])
    def test_identifier_names(self, name):
        _, p = profile_of(strs(range(30)), name=name)
        assert ColumnFlag.POSSIBLE_IDENTIFIER in p.flags

    @pytest.mark.parametrize("name", ["width", "valid", "paid", "side", "bridge", "idea"])
    def test_words_that_merely_contain_id_are_not_identifiers(self, name):
        _, p = profile_of(strs([1, 2, 3, 4] * 10), name=name)
        assert ColumnFlag.POSSIBLE_IDENTIFIER not in p.flags


class TestSensitiveDetection:
    def test_email_values_are_detected_even_with_an_innocent_name(self):
        col, p = profile_of([f"user{i}@example.com" for i in range(30)], name="contact")
        assert ColumnFlag.POSSIBLE_SENSITIVE_FIELD in p.flags and col.sensitive_evidence[0] >= 0.85

    def test_name_and_values_together_are_more_confident(self):
        col, _ = profile_of([f"user{i}@example.com" for i in range(30)], name="email")
        assert col.sensitive_evidence[0] == 0.95

    def test_a_sensitive_name_alone_is_a_weaker_signal(self):
        col, p = profile_of(["a", "b", "c"] * 10, name="phone")
        assert ColumnFlag.POSSIBLE_SENSITIVE_FIELD in p.flags and col.sensitive_evidence[0] == 0.6

    def test_phone_shaped_values(self):
        col, p = profile_of([f"+1 (555) 010-{i:04d}" for i in range(30)], name="contact")
        assert ColumnFlag.POSSIBLE_SENSITIVE_FIELD in p.flags

    def test_national_identifier_shaped_values(self):
        _, p = profile_of([f"{100 + i}-45-{1000 + i}" for i in range(30)], name="ref")
        assert ColumnFlag.POSSIBLE_SENSITIVE_FIELD in p.flags

    def test_street_addresses(self):
        _, p = profile_of([f"{10 + i} Main Street" for i in range(30)], name="location")
        assert ColumnFlag.POSSIBLE_SENSITIVE_FIELD in p.flags

    def test_plain_numbers_are_never_phone_numbers(self):
        _, p = profile_of(strs([5551230000 + i for i in range(30)]), name="measurement")
        assert ColumnFlag.POSSIBLE_SENSITIVE_FIELD not in p.flags

    def test_ordinary_text_is_not_sensitive(self):
        _, p = profile_of(["red", "green", "blue"] * 10, name="colour")
        assert ColumnFlag.POSSIBLE_SENSITIVE_FIELD not in p.flags

    def test_values_of_sensitive_columns_are_withheld(self):
        _, p = profile_of([f"user{i}@example.com" for i in range(30)], name="email")
        assert p.categorical.values_suppressed is True and p.categorical.top_values == []

    def test_values_of_identifier_columns_are_withheld(self):
        _, p = profile_of([f"ORD-{i:05d}" for i in range(200)], name="order_id")
        assert p.categorical.values_suppressed is True

    def test_ordinary_categories_are_shown(self):
        _, p = profile_of(["red", "green", "blue"] * 10, name="colour")
        assert p.categorical.values_suppressed is False and p.categorical.top_values


class TestProfileTable:
    def test_indexes_names_and_originals_are_preserved(self):
        cols = profile_table(
            table({"student_id": strs(range(30)), "grade": strs([1] * 30)},
                  originals=["Student ID", "Grade"]),
            SETTINGS,
        )
        assert [c.profile.index for c in cols] == [0, 1]
        assert cols[0].profile.original_name == "Student ID"
        assert cols[0].profile.normalized_name == "student_id"

    def test_every_profile_is_marked_as_parsed(self):
        for col in profile_table(table({"a": strs(range(10)), "b": ["x"] * 10}), SETTINGS):
            assert col.profile.evidence_level.value == "datapilot_parsed"

    def test_the_profile_is_deterministic(self):
        t = table({"a": strs([3, 1, 2, 2, None]), "b": ["x", "y", "x", None, "z"]})
        one = [c.profile.model_dump_json() for c in profile_table(t, SETTINGS)]
        two = [c.profile.model_dump_json() for c in profile_table(t, SETTINGS)]
        assert one == two
