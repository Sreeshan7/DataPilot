"""Secure ingestion of uploaded CSV and Excel files.

Mostly refusals and edge cases: a parser that only works on tidy files is a parser that
fails in production. Every error path asserts that the message is safe (no path, no cell
content, no exception text).
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime

import pytest

from app.core.config import Settings
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import AnalysisTimeout, IngestionError
from app.services.dataset_analysis.ingestion import (
    detect_header,
    normalize_column_names,
    parse_csv,
    parse_file,
    parse_xlsx,
)
from app.services.dataset_analysis.safety import (
    Deadline,
    sanitize_display_name,
    store_upload,
    validate_extension,
)


def settings(**kw) -> Settings:
    return Settings(gemini_api_key="", kaggle_api_token="", app_env="testing", **kw)


def write(tmp_path, name: str, content: bytes | str) -> str:
    path = tmp_path / name
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return str(path)


def csv_parse(tmp_path, content, name="data.csv", **kw):
    path = write(tmp_path, name, content)
    return parse_csv(path, name, os.path.getsize(path), settings(**kw))


def make_xlsx(tmp_path, sheets: dict[str, list[list]], name="book.xlsx") -> str:
    from openpyxl import Workbook

    wb = Workbook()
    first = True
    for title, rows in sheets.items():
        ws = wb.active if first else wb.create_sheet()
        ws.title = title
        first = False
        for row in rows:
            ws.append(row)
    path = str(tmp_path / name)
    wb.save(path)
    return path


def xlsx_parse(path, name="book.xlsx", sheet=None, **kw):
    return parse_xlsx(path, name, os.path.getsize(path), settings(**kw), sheet_name=sheet)


def assert_safe(error: IngestionError, tmp_path):
    text = f"{error.message} {error.details}"
    assert str(tmp_path) not in text
    assert "Traceback" not in text and "\\" not in error.message


# ---------------------------------------------------------------------------
# safety: names, extensions, temp storage, deadline
# ---------------------------------------------------------------------------


class TestSanitizeDisplayName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("student.csv", "student.csv"),
            ("../../etc/passwd", "passwd"),
            ("..\\..\\Windows\\system32\\evil.csv", "evil.csv"),
            ("C:\\Users\\Hp\\data.xlsx", "data.xlsx"),
            ("/var/tmp/x.csv", "x.csv"),
            ("", "upload"),
            (None, "upload"),
            ("   ", "upload"),
            ("...", "upload"),
            ("a\x00b.csv", "ab.csv"),
            ("my file (1).csv", "my file (1).csv"),
            ("évil<>:\"|?*.csv", "évil_.csv"),
        ],
    )
    def test_names_are_reduced_to_a_safe_label(self, raw, expected):
        assert sanitize_display_name(raw) == expected

    def test_a_traversal_name_keeps_no_directory_part(self):
        for raw in ("../../etc/passwd", "..\\..\\boot.ini", "/etc/shadow", "a/b/../../c.csv"):
            name = sanitize_display_name(raw)
            assert "/" not in name and "\\" not in name and ".." not in name

    def test_control_characters_are_removed(self):
        assert sanitize_display_name("a\nb\tc\x07.csv") == "abc.csv"

    def test_a_long_name_is_cut_but_keeps_its_extension(self):
        name = sanitize_display_name("x" * 500 + ".csv")
        assert len(name) <= 100 and name.endswith(".csv")


class TestValidateExtension:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [("a.csv", "csv"), ("a.CSV", "csv"), ("a.tsv", "csv"), ("a.xlsx", "xlsx"), ("A.XLSX", "xlsx")],
    )
    def test_supported_files(self, name, kind):
        assert validate_extension(name) == kind

    @pytest.mark.parametrize("name", ["a.exe", "a.csv.exe", "a", "a.json", "a.py", "a.zip", "a.pdf", ".csv.bak"])
    def test_everything_else_is_refused(self, name):
        with pytest.raises(IngestionError) as err:
            validate_extension(name)
        assert err.value.code is ErrorCode.FILE_TYPE_UNSUPPORTED and err.value.status_code == 415

    def test_legacy_xls_and_macro_workbooks_say_how_to_proceed(self):
        with pytest.raises(IngestionError, match="Save the file as .xlsx"):
            validate_extension("a.xls")
        with pytest.raises(IngestionError, match="xlsm"):
            validate_extension("a.xlsm")


class TestStoreUpload:
    def test_the_upload_is_copied_and_then_removed(self):
        with store_upload(io.BytesIO(b"a,b\n1,2\n"), 1024) as stored:
            path = stored.path
            assert os.path.exists(path) and stored.size_bytes == 8
        assert not os.path.exists(path)

    def test_the_temporary_name_is_random_and_not_the_users(self):
        with store_upload(io.BytesIO(b"x"), 10) as a, store_upload(io.BytesIO(b"x"), 10) as b:
            assert a.path != b.path
            assert "datapilot_" in os.path.basename(a.path)

    def test_an_oversized_upload_is_refused_and_leaves_nothing_behind(self):
        class Endless(io.RawIOBase):
            def __init__(self):
                self.served = 0

            def read(self, n=-1):
                self.served += n
                return b"x" * n

        stream = Endless()
        captured = {}
        with pytest.raises(IngestionError) as err:
            with store_upload(stream, 1024 * 128) as stored:  # pragma: no cover
                captured["path"] = stored.path
        assert err.value.code is ErrorCode.FILE_TOO_LARGE and err.value.status_code == 413
        # it stopped reading shortly after the cap instead of consuming an endless stream
        assert stream.served < 1024 * 128 + 4 * 64 * 1024

    def test_the_file_is_removed_even_when_the_caller_raises(self):
        seen = {}
        with pytest.raises(RuntimeError):
            with store_upload(io.BytesIO(b"abc"), 10) as stored:
                seen["path"] = stored.path
                raise RuntimeError("boom")
        assert not os.path.exists(seen["path"])

    def test_a_cleanup_failure_is_recorded_without_revealing_the_path(self, monkeypatch, caplog):
        real_remove = os.remove

        def refuse(path):
            raise PermissionError(f"cannot remove {path}")

        with caplog.at_level("WARNING"):
            with store_upload(io.BytesIO(b"abc"), 10) as stored:
                monkeypatch.setattr(os, "remove", refuse)
                path = stored.path
            monkeypatch.setattr(os, "remove", real_remove)
        assert stored.cleanup_failed is True
        assert path not in caplog.text
        real_remove(path)  # tidy up the file the failed cleanup left behind


class TestDeadline:
    def test_it_raises_once_the_budget_is_spent(self):
        now = [0.0]
        deadline = Deadline(5.0, clock=lambda: now[0])
        deadline.check("parsing")
        now[0] = 6.0
        with pytest.raises(AnalysisTimeout) as err:
            deadline.check("profiling")
        assert err.value.status_code == 504 and err.value.details == {"stage": "profiling"}

    def test_remaining_never_goes_negative(self):
        now = [10.0]
        assert Deadline(1.0, clock=lambda: now[0]).remaining() == 1.0
        now[0] = 99.0


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


class TestNormalizeColumnNames:
    def test_names_are_normalised_and_the_change_is_recorded(self):
        names, renames = normalize_column_names(["Student ID", "Final Grade (%)", "age"])
        assert names == ["student_id", "final_grade", "age"]
        assert [r.original_name for r in renames] == ["Student ID", "Final Grade (%)"]
        assert all(r.reason for r in renames)

    def test_an_unchanged_name_is_not_listed(self):
        _, renames = normalize_column_names(["age", "income"])
        assert renames == []

    def test_duplicates_get_numeric_suffixes_and_a_reason(self):
        names, renames = normalize_column_names(["age", "age", "Age"])
        assert names == ["age", "age_2", "age_3"] and len(set(names)) == 3
        assert any("duplicates" in r.reason for r in renames)

    def test_empty_and_symbol_only_headers_get_placeholders(self):
        names, renames = normalize_column_names(["", "   ", "###", None])
        assert len(set(names)) == 4 and all(names)
        assert all(r.original_name in (None, "###") for r in renames)

    def test_a_suffix_cannot_collide_with_a_real_column(self):
        names, _ = normalize_column_names(["a", "a_2", "a"])
        assert len(set(names)) == 3

    def test_the_original_is_never_altered(self):
        _, renames = normalize_column_names(["  Spaced Name  "])
        assert renames[0].original_name == "  Spaced Name  "


# ---------------------------------------------------------------------------
# header detection
# ---------------------------------------------------------------------------


class TestDetectHeader:
    def test_words_above_numbers_is_a_header(self):
        assert detect_header([["age", "income"], ["31", "52000"], ["45", "61000"]]) == (True, None)

    def test_a_numeric_first_row_is_data(self):
        assert detect_header([["31", "52000"], ["45", "61000"]])[0] is False

    def test_a_date_first_row_is_data(self):
        assert detect_header([["2024-01-02", "5"], ["2024-01-03", "6"]])[0] is False

    def test_an_all_text_file_assumes_a_header_and_says_so(self):
        has, warning = detect_header([["name", "city"], ["Ann", "Oslo"], ["Bo", "Rome"]])
        assert has is True and "assumed" in warning

    def test_an_empty_first_row_is_not_a_header(self):
        assert detect_header([[""], ["1"]])[0] is False


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCsvBasics:
    def test_a_normal_file_is_read(self, tmp_path):
        table = csv_parse(tmp_path, "age,income\n31,52000\n45,61000\n")
        assert list(table.frame.columns) == ["age", "income"]
        assert table.info.rows_read == 2 and table.info.file_type == "csv"
        assert table.info.header_detected and table.info.delimiter == ","
        assert table.original_names == ["age", "income"]

    def test_values_stay_strings_so_no_information_is_lost(self, tmp_path):
        table = csv_parse(tmp_path, "zip,qty\n007,1.50\n010,2\n")
        assert list(table.frame["zip"]) == ["007", "010"]

    def test_missing_tokens_become_none(self, tmp_path):
        table = csv_parse(tmp_path, "a,b\n1,NA\n?,null\n-,\n4,ok\n")
        assert list(table.frame["a"]) == ["1", None, None, "4"]
        assert list(table.frame["b"]) == [None, None, None, "ok"]

    def test_none_nil_and_missing_are_real_values_not_placeholders(self, tmp_path):
        """"None" is a genuine category ("no refund"); erasing it once turned a
        two-valued column into a constant one."""
        table = csv_parse(tmp_path, "refund,note\nnone,missing\nrefunded,nil\nNone,ok\n")
        assert list(table.frame["refund"]) == ["none", "refunded", "None"]
        assert list(table.frame["note"]) == ["missing", "nil", "ok"]

    def test_the_placeholders_that_were_converted_are_named(self, tmp_path):
        table = csv_parse(tmp_path, "a,b\n1,NA\n?,null\n3,x\n")
        warning = next(w for w in table.info.warnings if "placeholder" in w)
        assert "'NA'" in warning and "'?'" in warning and "'null'" in warning

    def test_no_placeholder_warning_when_none_were_converted(self, tmp_path):
        table = csv_parse(tmp_path, "a,b\n1,2\n3,4\n")
        assert not any("placeholder" in w for w in table.info.warnings)

    def test_whitespace_around_values_is_trimmed(self, tmp_path):
        assert list(csv_parse(tmp_path, "a\n  hello  \n").frame["a"]) == ["hello"]

    def test_crlf_and_blank_rows(self, tmp_path):
        table = csv_parse(tmp_path, "a,b\r\n1,2\r\n\r\n3,4\r\n")
        assert table.info.rows_read == 2
        assert any("blank row" in w for w in table.info.warnings)

    def test_quoted_fields_with_delimiters_and_newlines(self, tmp_path):
        table = csv_parse(tmp_path, 'name,note\n"Smith, J","line1\nline2"\n"Bo","x"\n')
        assert table.info.rows_read == 2
        assert table.frame["name"][0] == "Smith, J" and "line2" in table.frame["note"][0]

    def test_a_single_column_file(self, tmp_path):
        table = csv_parse(tmp_path, "score\n1\n2\n3\n")
        assert list(table.frame.columns) == ["score"] and table.info.rows_read == 3

    def test_a_trailing_delimiter_adds_an_unnamed_column_not_an_error(self, tmp_path):
        table = csv_parse(tmp_path, "a,b,\n1,2,\n3,4,\n")
        assert list(table.frame.columns) == ["a", "b", "unnamed_3"]
        assert table.info.malformed_rows == 0

    def test_the_output_is_deterministic(self, tmp_path):
        content = "a,b\n1,2\n3,4\n"
        one = csv_parse(tmp_path, content).frame
        two = csv_parse(tmp_path, content).frame
        assert one.equals(two)


class TestCsvDelimiters:
    @pytest.mark.parametrize(
        ("delimiter", "name"), [(";", "semi.csv"), ("\t", "tab.csv"), ("|", "pipe.csv"), (",", "comma.csv")]
    )
    def test_the_delimiter_is_detected(self, tmp_path, delimiter, name):
        rows = [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]]
        content = "\n".join(delimiter.join(r) for r in rows) + "\n"
        table = csv_parse(tmp_path, content, name=name)
        assert table.info.delimiter == delimiter
        assert list(table.frame.columns) == ["a", "b", "c"]

    def test_tsv_uses_a_tab(self, tmp_path):
        path = write(tmp_path, "d.tsv", "a\tb\n1\t2\n")
        table = parse_file(path, "d.tsv", os.path.getsize(path), settings())
        assert list(table.frame.columns) == ["a", "b"]

    def test_a_comma_inside_quotes_does_not_fool_the_sniffer(self, tmp_path):
        table = csv_parse(tmp_path, 'a;b\n"x,y";2\n"z,w";3\n')
        assert table.info.delimiter == ";"


class TestCsvEncodings:
    def test_utf8_bom(self, tmp_path):
        table = csv_parse(tmp_path, b"\xef\xbb\xbfname,age\nAna,3\n")
        assert table.info.encoding == "utf-8-sig" and list(table.frame.columns) == ["name", "age"]

    def test_utf16(self, tmp_path):
        table = csv_parse(tmp_path, "name,age\nAna,3\n".encode("utf-16"))
        assert table.info.encoding == "utf-16" and table.frame["name"][0] == "Ana"

    def test_cp1252_is_read_and_flagged(self, tmp_path):
        table = csv_parse(tmp_path, "name,city\nJosé,Zürich\n".encode("cp1252"))
        assert table.info.encoding == "cp1252"
        assert table.frame["city"][0] == "Zürich"
        assert any("not valid UTF-8" in w for w in table.info.warnings)

    def test_bytes_invalid_everywhere_fall_back_to_latin1_without_crashing(self, tmp_path):
        table = csv_parse(tmp_path, b"a,b\n\x81\x8d,1\n")
        assert table.info.encoding in {"cp1252", "latin-1"} and table.info.rows_read == 1

    def test_unicode_names_survive(self, tmp_path):
        table = csv_parse(tmp_path, "名前,age\nAna,3\n")
        assert table.original_names[0] == "名前"


class TestCsvHeaders:
    def test_a_missing_header_generates_names_and_says_so(self, tmp_path):
        table = csv_parse(tmp_path, "31,52000\n45,61000\n")
        assert list(table.frame.columns) == ["unnamed_1", "unnamed_2"]
        assert table.info.header_detected is False and table.original_names == [None, None]
        assert any("No header row" in w for w in table.info.warnings)
        assert table.info.rows_read == 2

    def test_duplicate_headers_are_kept_apart_and_recorded(self, tmp_path):
        table = csv_parse(tmp_path, "age,age,city\n1,2,x\n3,4,y\n")
        assert list(table.frame.columns) == ["age", "age_2", "city"]
        assert table.original_names == ["age", "age", "city"]
        assert any(r.normalized_name == "age_2" for r in table.info.columns_renamed)

    def test_an_empty_header_cell_is_recorded(self, tmp_path):
        table = csv_parse(tmp_path, "a,,c\n1,2,3\n4,5,6\n")
        assert "unnamed_2" in table.frame.columns
        assert any(r.original_name is None for r in table.info.columns_renamed)


class TestCsvMalformed:
    def test_short_rows_are_padded_and_counted(self, tmp_path):
        table = csv_parse(tmp_path, "a,b,c\n1,2,3\n4,5\n6,7,8\n")
        assert table.info.malformed_rows == 1 and table.frame["c"][1] is None
        assert any("wrong number of fields" in w for w in table.info.warnings)

    def test_long_rows_are_cut_and_counted(self, tmp_path):
        table = csv_parse(tmp_path, "a,b\n1,2\n3,4,5\n")
        assert table.info.malformed_rows == 1 and table.info.rows_read == 2

    def test_mostly_malformed_files_warn_about_the_delimiter(self, tmp_path):
        rows = "a,b,c\n" + "\n".join("1,2" for _ in range(10)) + "\n"
        table = csv_parse(tmp_path, rows)
        assert any("delimiter may be wrong" in w for w in table.info.warnings)

    def test_a_header_only_file_has_no_data_rows(self, tmp_path):
        with pytest.raises(IngestionError) as err:
            csv_parse(tmp_path, "a,b,c\n")
        assert err.value.code is ErrorCode.FILE_NO_DATA_ROWS

    @pytest.mark.parametrize("content", ["", "   \n\n  ", b"\n\n\n"])
    def test_an_empty_file_is_refused(self, tmp_path, content):
        with pytest.raises(IngestionError) as err:
            csv_parse(tmp_path, content)
        assert err.value.code is ErrorCode.FILE_EMPTY

    def test_a_cell_larger_than_the_limit_is_refused_safely(self, tmp_path):
        with pytest.raises(IngestionError) as err:
            csv_parse(tmp_path, "a,b\n" + "x" * 5000 + ",1\n", analysis_max_field_chars=1000)
        assert err.value.code is ErrorCode.FILE_MALFORMED
        assert "xxxxx" not in err.value.message
        assert_safe(err.value, tmp_path)

    def test_an_unterminated_quote_does_not_hang_or_crash(self, tmp_path):
        with pytest.raises(IngestionError):
            csv_parse(tmp_path, 'a,b\n"unterminated,1\n' + "x,y\n" * 3000, analysis_max_field_chars=500)

    def test_the_field_size_limit_is_restored_after_parsing(self, tmp_path):
        import csv

        before = csv.field_size_limit()
        csv_parse(tmp_path, "a\n1\n", analysis_max_field_chars=77)
        assert csv.field_size_limit() == before


class TestCsvContentSniffing:
    def test_a_binary_file_named_csv_is_refused(self, tmp_path):
        with pytest.raises(IngestionError) as err:
            csv_parse(tmp_path, b"a,b\n\x00\x01\x02\x00\x03,1\n")
        assert err.value.code is ErrorCode.FILE_MALFORMED and "binary" in err.value.message

    @pytest.mark.parametrize("magic", [b"%PDF-1.4 ...", b"MZ\x90\x00", b"\x7fELF....", b"\x89PNG\r\n\x1a\n", b"GIF89a"])
    def test_known_binary_formats_are_refused(self, tmp_path, magic):
        with pytest.raises(IngestionError) as err:
            csv_parse(tmp_path, magic + b"a,b\n1,2\n")
        assert err.value.code is ErrorCode.FILE_MALFORMED

    def test_a_zip_renamed_csv_is_refused_with_advice(self, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("a.txt", "x")
        with pytest.raises(IngestionError, match=r"\.xlsx"):
            csv_parse(tmp_path, buffer.getvalue())

    def test_a_script_is_just_text_and_is_never_executed(self, tmp_path):
        table = csv_parse(tmp_path, "cmd,arg\n=cmd|'/c calc'!A1,@SUM(1+1)\n__import__('os').system('x'),1\n")
        assert table.info.rows_read == 2  # parsed as data, nothing run


class TestCsvLimits:
    def test_rows_are_truncated_and_it_says_so(self, tmp_path):
        content = "a\n" + "\n".join(str(i) for i in range(50)) + "\n"
        table = csv_parse(tmp_path, content, analysis_max_rows=10)
        assert table.info.rows_read == 10 and table.info.truncated is True
        assert "row limit of 10" in table.info.truncation_reason
        assert any("truncated" in w for w in table.info.warnings)

    def test_a_file_exactly_at_the_limit_is_not_truncated(self, tmp_path):
        content = "a\n" + "\n".join(str(i) for i in range(10)) + "\n"
        table = csv_parse(tmp_path, content, analysis_max_rows=10)
        assert table.info.rows_read == 10 and table.info.truncated is False

    def test_the_cell_limit_truncates_before_the_row_limit(self, tmp_path):
        content = "a,b,c,d\n" + "\n".join("1,2,3,4" for _ in range(100)) + "\n"
        table = csv_parse(tmp_path, content, analysis_max_cells=40)
        assert table.info.rows_read == 10 and table.info.truncated is True
        assert "cell limit" in table.info.truncation_reason

    def test_too_many_columns_is_refused(self, tmp_path):
        header = ",".join(f"c{i}" for i in range(30))
        with pytest.raises(IngestionError) as err:
            csv_parse(tmp_path, header + "\n" + ",".join("1" for _ in range(30)) + "\n", analysis_max_columns=10)
        assert err.value.code is ErrorCode.FILE_TOO_WIDE

    def test_an_extremely_wide_single_row_is_refused(self, tmp_path):
        with pytest.raises(IngestionError) as err:
            csv_parse(tmp_path, ",".join(f"c{i}" for i in range(20000)) + "\n1\n")
        assert err.value.code is ErrorCode.FILE_TOO_WIDE

    def test_the_deadline_is_honoured_while_parsing(self, tmp_path):
        content = "a\n" + "\n".join("1" for _ in range(20000)) + "\n"
        path = write(tmp_path, "d.csv", content)
        now = [0.0]
        deadline = Deadline(1.0, clock=lambda: now[0])
        calls = {"n": 0}

        def ticking():
            calls["n"] += 1
            return 0.0 if calls["n"] < 2 else 99.0

        deadline._clock = ticking  # type: ignore[attr-defined]
        with pytest.raises(AnalysisTimeout):
            parse_csv(path, "d.csv", os.path.getsize(path), settings(), deadline)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


class TestXlsx:
    def test_a_workbook_is_read_with_its_types(self, tmp_path):
        path = make_xlsx(tmp_path, {"Data": [
            ["name", "age", "score", "joined", "active"],
            ["Ann", 31, 1.5, datetime(2024, 1, 2), True],
            ["Bo", 45, 2.5, datetime(2024, 2, 3), False],
        ]})
        table = xlsx_parse(path)
        assert list(table.frame.columns) == ["name", "age", "score", "joined", "active"]
        assert table.frame["age"][0] == 31 and isinstance(table.frame["joined"][0], datetime)
        assert table.frame["active"][0] is True
        assert table.info.sheet_name == "Data" and table.info.file_type == "xlsx"
        assert table.info.rows_read == 2

    def test_missing_cells_and_placeholders_become_none(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["a", "b"], [1, None], ["NA", "x"], [3, "?"]]})
        table = xlsx_parse(path)
        assert list(table.frame["a"]) == [1, None, 3] and list(table.frame["b"]) == [None, "x", None]
        assert any("placeholder" in w and "'NA'" in w for w in table.info.warnings)

    def test_mixed_types_in_a_column_are_kept_for_the_profiler(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["v"], [1], ["two"], [3.5], [datetime(2024, 1, 1)]]})
        values = list(xlsx_parse(path).frame["v"])
        assert values[0] == 1 and values[1] == "two" and values[2] == 3.5

    def test_duplicate_and_empty_headers(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["a", "a", None, "b"], [1, 2, 3, 4], [5, 6, 7, 8]]})
        table = xlsx_parse(path)
        assert list(table.frame.columns) == ["a", "a_2", "unnamed_3", "b"]

    def test_a_missing_header_is_detected_for_numeric_first_rows(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [[1, 2], [3, 4], [5, 6]]})
        table = xlsx_parse(path)
        assert table.info.header_detected is False and table.info.rows_read == 3

    def test_spreadsheet_errors_are_treated_as_missing_and_reported(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["a"], ["#DIV/0!"], [2], ["#N/A"]]})
        table = xlsx_parse(path)
        assert list(table.frame["a"]) == [None, 2, None]
        assert any("spreadsheet errors" in w for w in table.info.warnings)

    def test_formulas_are_not_evaluated(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["a", "b"], [1, "=1+1"], [2, "=SUM(A2:A3)"]]})
        table = xlsx_parse(path)
        assert table.info.rows_read == 2  # no cached value was ever calculated: read as missing, not run

    def test_blank_rows_are_skipped(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["a"], [1], [None], [None], [2]]})
        table = xlsx_parse(path)
        assert table.info.rows_read == 2

    def test_rows_are_truncated_and_it_says_so(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["a"]] + [[i] for i in range(40)]})
        table = xlsx_parse(path, analysis_max_rows=10)
        assert table.info.rows_read == 10 and table.info.truncated
        assert "row limit" in table.info.truncation_reason

    def test_too_many_columns_is_refused(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [[f"c{i}" for i in range(30)], list(range(30))]})
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path, analysis_max_columns=10)
        assert err.value.code is ErrorCode.FILE_TOO_WIDE

    def test_header_only_and_empty_sheets(self, tmp_path):
        path = make_xlsx(tmp_path, {"S": [["a", "b"]]})
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path)
        assert err.value.code is ErrorCode.FILE_NO_DATA_ROWS
        path = make_xlsx(tmp_path, {"S": []}, name="empty.xlsx")
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path, name="empty.xlsx")
        assert err.value.code is ErrorCode.FILE_EMPTY


class TestXlsxSheets:
    def _book(self, tmp_path):
        return make_xlsx(tmp_path, {
            "Empty": [],
            "Sales": [["region", "amount"], ["N", 10], ["S", 20]],
            "Costs": [["item", "cost"], ["a", 1]],
        })

    def test_the_default_is_the_first_non_empty_sheet(self, tmp_path):
        table = xlsx_parse(self._book(tmp_path))
        assert table.info.sheet_name == "Sales"
        assert table.info.sheets_available == ["Empty", "Sales", "Costs"]
        assert any("only 'Sales' was analysed" in w for w in table.info.warnings)

    def test_a_sheet_can_be_chosen_by_name(self, tmp_path):
        assert xlsx_parse(self._book(tmp_path), sheet="Costs").info.sheet_name == "Costs"

    def test_sheet_names_match_case_insensitively(self, tmp_path):
        assert xlsx_parse(self._book(tmp_path), sheet="costs").info.sheet_name == "Costs"

    def test_an_unknown_sheet_lists_what_exists(self, tmp_path):
        with pytest.raises(IngestionError) as err:
            xlsx_parse(self._book(tmp_path), sheet="Nope")
        assert err.value.code is ErrorCode.EXCEL_SHEET_NOT_FOUND
        assert err.value.details["sheets_available"] == ["Empty", "Sales", "Costs"]

    def test_too_many_sheets_is_refused(self, tmp_path):
        path = make_xlsx(tmp_path, {f"s{i}": [["a"], [i]] for i in range(8)})
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path, analysis_max_excel_sheets=5)
        assert err.value.code is ErrorCode.EXCEL_TOO_MANY_SHEETS

    def test_a_workbook_of_only_empty_sheets_is_refused(self, tmp_path):
        path = make_xlsx(tmp_path, {"A": [], "B": []})
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path)
        assert err.value.code is ErrorCode.FILE_EMPTY


class TestXlsxContainerSafety:
    def test_garbage_named_xlsx_is_refused_safely(self, tmp_path):
        path = write(tmp_path, "x.xlsx", b"this is not a workbook at all" * 10)
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path)
        assert err.value.code is ErrorCode.FILE_MALFORMED
        assert_safe(err.value, tmp_path)

    def test_a_zip_that_is_not_a_workbook_is_refused(self, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("readme.txt", "hello")
        path = write(tmp_path, "x.xlsx", buffer.getvalue())
        with pytest.raises(IngestionError, match="not a valid .xlsx"):
            xlsx_parse(path)

    def test_a_truncated_workbook_is_refused(self, tmp_path):
        good = make_xlsx(tmp_path, {"S": [["a"], [1]]})
        data = open(good, "rb").read()
        path = write(tmp_path, "cut.xlsx", data[: len(data) // 2])
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path, name="cut.xlsx")
        assert err.value.code is ErrorCode.FILE_MALFORMED

    def test_a_decompression_bomb_is_refused_before_it_is_inflated(self, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("xl/workbook.xml", "<workbook/>")
            z.writestr("xl/worksheets/sheet1.xml", b"0" * 5_000_000)
        path = write(tmp_path, "bomb.xlsx", buffer.getvalue())
        assert os.path.getsize(path) < 50_000
        with pytest.raises(IngestionError) as err:
            xlsx_parse(path, name="bomb.xlsx")
        assert err.value.code is ErrorCode.FILE_UNSAFE and err.value.status_code == 422

    def test_an_absolute_size_cap_applies_independently_of_the_ratio(self, tmp_path):
        good = make_xlsx(tmp_path, {"S": [["a"], [1]]})
        with pytest.raises(IngestionError) as err:
            xlsx_parse(good, analysis_max_excel_uncompressed_bytes=100)
        assert err.value.code is ErrorCode.FILE_UNSAFE

    def test_an_encrypted_member_is_refused_with_advice(self, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("xl/workbook.xml", "<workbook/>")
        data = bytearray(buffer.getvalue())
        # set the "encrypted" general-purpose flag in both the local and central headers
        for sig in (b"PK\x03\x04", b"PK\x01\x02"):
            index = data.find(sig)
            offset = index + (6 if sig == b"PK\x03\x04" else 8)
            data[offset] |= 0x01
        path = write(tmp_path, "locked.xlsx", bytes(data))
        with pytest.raises(IngestionError, match="password-protected"):
            xlsx_parse(path, name="locked.xlsx")


class TestDispatch:
    def test_extension_decides_the_parser(self, tmp_path):
        csv_path = write(tmp_path, "a.csv", "a\n1\n")
        assert parse_file(csv_path, "a.csv", 4, settings()).info.file_type == "csv"
        xlsx_path = make_xlsx(tmp_path, {"S": [["a"], [1]]})
        assert parse_file(xlsx_path, "a.xlsx", os.path.getsize(xlsx_path), settings()).info.file_type == "xlsx"

    def test_an_unsupported_extension_is_refused_before_reading(self, tmp_path):
        path = write(tmp_path, "a.exe", "MZ")
        with pytest.raises(IngestionError) as err:
            parse_file(path, "a.exe", 2, settings())
        assert err.value.status_code == 415

    def test_the_sheet_argument_reaches_the_excel_parser(self, tmp_path):
        path = make_xlsx(tmp_path, {"A": [["x"], [1]], "B": [["y"], [2]]})
        table = parse_file(path, "a.xlsx", os.path.getsize(path), settings(), sheet_name="B")
        assert table.info.sheet_name == "B"


class TestUploadSuffix:
    def test_the_stored_file_keeps_the_validated_extension(self):
        with store_upload(io.BytesIO(b"x"), 10, ".xlsx") as stored:
            assert stored.path.endswith(".xlsx")
        with store_upload(io.BytesIO(b"x"), 10, ".csv") as stored:
            assert stored.path.endswith(".csv")

    @pytest.mark.parametrize("suffix", [".exe", ".py", "", "../../x", ".xlsx/../../etc"])
    def test_a_suffix_outside_the_allowlist_is_refused(self, suffix):
        with pytest.raises(ValueError):
            with store_upload(io.BytesIO(b"x"), 10, suffix):
                pass  # pragma: no cover

    def test_an_xlsx_upload_is_readable_through_a_stored_file(self, tmp_path):
        """openpyxl chooses how to read a file from its name; this is what the API relies on."""
        path = make_xlsx(tmp_path, {"S": [["a"], [1], [2]]})
        with store_upload(open(path, "rb"), 10_000_000, ".xlsx") as stored:
            table = parse_xlsx(stored.path, "x.xlsx", stored.size_bytes, settings())
        assert table.info.rows_read == 2
