"""Targets and tasks for a dataset known only from a registry's published schema.

The rules under test: with no rows read, a target is STRONG only if the registry designates it,
a name alone caps at POSSIBLE, no evidence is ever marked as parsed, and a task the registry
cannot support (unknown class count, no declared type) says what is unknown instead of guessing.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset_analysis import (
    CandidateState,
    EvidenceLevel,
    ExternalMetadata,
    MLTask,
    SchemaColumn,
    TargetKind,
)
from app.services.dataset_analysis.external.service import _distinct_alternatives
from app.services.dataset_analysis.external.adapters import ExternalHit
from app.services.dataset_analysis.external_ml import detect_external_targets, detect_external_tasks

REGISTRY = EvidenceLevel.REGISTRY_VERIFIED


def col(name, dtype=None, role=None, description=None):
    return SchemaColumn(original_name=name, normalized_name=name.lower().replace(" ", "_"), data_type=dtype,
                        role=role, description=description, evidence=REGISTRY)


def meta(columns, targets=(), tasks=(), stats=None, rows=None):
    return ExternalMetadata(
        source="uci", identifier="1", name="X", schema_columns=columns, schema_level=REGISTRY,
        schema_source="test", registry_target_columns=list(targets), tasks=list(tasks),
        registry_statistics=stats or {}, row_count=rows)


class TestExternalTargets:
    def test_a_registry_designated_target_is_strong_with_registry_evidence(self):
        m = meta([col("a", "Continuous"), col("outcome_flag", "Binary", "target")], targets=["outcome_flag"])
        best = detect_external_targets(m)[0]
        assert best.column == "outcome_flag" and best.state is CandidateState.STRONG_CANDIDATE
        assert best.target_kind is TargetKind.BINARY
        assert any(e.level is REGISTRY and "designates" in e.statement for e in best.evidence)

    def test_a_name_alone_never_makes_a_target_strong(self):
        m = meta([col("age", "Integer"), col("income", "Integer"), col("churn", "Binary")])
        best = detect_external_targets(m)[0]
        assert best.column == "churn" and best.state is CandidateState.POSSIBLE
        assert best.confidence <= 0.6

    def test_no_candidate_is_ever_marked_as_parsed(self):
        m = meta([col("a"), col("b"), col("label", "Binary", "target")], targets=["label"])
        for cand in detect_external_targets(m):
            assert all(e.level is not EvidenceLevel.DATAPILOT_PARSED for e in cand.evidence)

    def test_every_candidate_says_no_rows_were_read(self):
        m = meta([col("a"), col("target", "Binary")])
        assert all(any("no rows were read" in a for a in c.against) for c in detect_external_targets(m))

    def test_a_row_identifier_or_ignored_column_is_penalised(self):
        m = meta([col("id", "Integer", "row_id"), col("y", "Binary", "target")], targets=["y"])
        assert [c.column for c in detect_external_targets(m)][0] == "y"
        assert "id" not in {c.column for c in detect_external_targets(m)}

    def test_no_schema_means_no_candidates(self):
        assert detect_external_targets(ExternalMetadata(source="kaggle", identifier="a/b", name="B")) == []

    def test_a_user_named_target_is_strong_and_user_specified(self):
        m = meta([col("age", "Integer"), col("weight", "Continuous")])
        best = detect_external_targets(m, "Weight")[0]
        assert best.column == "weight" and best.user_specified and best.confidence == 1.0

    def test_a_registry_target_is_found_by_its_original_or_normalised_name(self):
        m = meta([col("Sepal Length", "Continuous"), col("Class", "Categorical", "target")], targets=["Class"])
        assert detect_external_targets(m)[0].column == "class"

    @pytest.mark.parametrize(
        ("dtype", "stats", "kind"),
        [("Binary", {}, TargetKind.BINARY), ("Categorical", {"NumberOfClasses": 2.0}, TargetKind.BINARY),
         ("Categorical", {"NumberOfClasses": 5.0}, TargetKind.MULTICLASS), ("Categorical", {}, TargetKind.UNKNOWN),
         ("Continuous", {}, TargetKind.CONTINUOUS), ("Integer", {}, TargetKind.UNKNOWN), (None, {}, TargetKind.UNKNOWN)],
    )
    def test_the_kind_comes_only_from_what_the_registry_published(self, dtype, stats, kind):
        m = meta([col("a"), col("y", dtype, "target")], targets=["y"], stats=stats)
        assert detect_external_targets(m)[0].target_kind is kind

    def test_a_class_label_with_a_published_class_count_is_multiclass(self):
        m = meta([col("image", "image"), col("label", "class_label", description="10 classes")], targets=["label"])
        assert detect_external_targets(m)[0].target_kind is TargetKind.MULTICLASS
        two = meta([col("x"), col("label", "class_label", description="2 classes")], targets=["label"])
        assert detect_external_targets(two)[0].target_kind is TargetKind.BINARY


class TestExternalTasks:
    def tasks(self, m, hint=None):
        return detect_external_tasks(m, detect_external_targets(m, hint))

    def test_binary_classification_from_a_binary_registry_target(self):
        m = meta([col("a"), col("y", "Binary", "target")], targets=["y"])
        task = self.tasks(m)[0]
        assert task.task is MLTask.BINARY_CLASSIFICATION and task.target == "y"

    def test_regression_from_a_continuous_registry_target(self):
        m = meta([col("a"), col("price", "Continuous", "target")], targets=["price"])
        assert self.tasks(m)[0].task is MLTask.REGRESSION

    def test_a_declared_classification_task_covers_an_integer_coded_target(self):
        """UCI lists a coded diagnosis as Integer; the registry still declares classification."""
        m = meta([col("age", "Integer"), col("num", "Integer", "target")], targets=["num"], tasks=["Classification"])
        task = self.tasks(m)[0]
        assert task.task is MLTask.MULTICLASS_CLASSIFICATION and task.target == "num"
        assert any("declares classification" in e.statement for e in task.evidence)
        assert any("number of classes was not published" in l for l in task.limitations)

    def test_a_declared_regression_task_covers_an_untyped_target(self):
        m = meta([col("a"), col("y", None, "target")], targets=["y"], tasks=["Regression"])
        assert self.tasks(m)[0].task is MLTask.REGRESSION

    def test_time_series_needs_the_registry_to_say_so(self):
        plain = meta([col("a"), col("v", "Continuous", "target")], targets=["v"])
        declared = meta([col("a"), col("v", "Continuous", "target")], targets=["v"], tasks=["Time Series"])
        assert MLTask.TIME_SERIES_FORECASTING not in {t.task for t in self.tasks(plain)}
        assert MLTask.TIME_SERIES_FORECASTING in {t.task for t in self.tasks(declared)}

    def test_clustering_comes_from_a_declared_task_and_has_no_target(self):
        m = meta([col("a"), col("b")], tasks=["Clustering"])
        task = self.tasks(m)[0]
        assert task.task is MLTask.CLUSTERING and task.target is None

    def test_no_schema_and_no_declared_task_means_no_task(self):
        assert detect_external_tasks(ExternalMetadata(source="kaggle", identifier="a", name="A"), []) == []

    def test_tasks_never_claim_more_than_possible_without_rows(self):
        m = meta([col("a"), col("y", "Binary", "target")], targets=["y"])
        for task in self.tasks(m):
            assert task.state in (CandidateState.POSSIBLE, CandidateState.WEAK)
            assert any("no rows were read" in l for l in task.limitations)

    def test_an_unpublished_row_count_is_listed_as_unknown(self):
        m = meta([col("a"), col("y", "Binary", "target")], targets=["y"], rows=None)
        assert any("number of rows" in l for l in self.tasks(m)[0].limitations)

    def test_a_published_row_count_is_not_called_unknown(self):
        m = meta([col("a"), col("y", "Binary", "target")], targets=["y"], rows=500)
        assert not any("number of rows" in l for l in self.tasks(m)[0].limitations)

    def test_every_supervised_task_targets_a_real_column(self):
        m = meta([col("a"), col("b", "Continuous"), col("y", "Binary", "target")], targets=["y"], tasks=["Classification"])
        names = {c.normalized_name for c in m.schema_columns}
        assert all(t.target in names for t in self.tasks(m) if t.target)


class TestDistinctAlternatives:
    class A:
        authority = 1

    def hit(self, source, ident, name, cols=None, rows=None):
        return (self.A(), ExternalHit(source=source, identifier=ident, name=name, column_count=cols, row_count=rows))

    def test_openml_versions_with_the_same_shape_collapse_to_one(self):
        others = [self.hit("openml", str(i), f"iris (v{i})", 5, 150) for i in range(26, 0, -1)]
        assert len(_distinct_alternatives(others)) == 1

    def test_openml_versions_with_a_different_shape_stay_separate(self):
        others = [self.hit("openml", "1", "iris (v1)", 5, 150), self.hit("openml", "2", "iris (v2)", 4, 150)]
        assert len(_distinct_alternatives(others)) == 2

    def test_kaggle_datasets_with_the_same_title_but_different_owners_are_not_merged(self):
        others = [self.hit("kaggle", "a/titanic", "Titanic"), self.hit("kaggle", "b/titanic", "Titanic")]
        assert [a.identifier for a in _distinct_alternatives(others)] == ["a/titanic", "b/titanic"]

    def test_the_list_is_capped(self):
        others = [self.hit("kaggle", f"o/{i}", "Titanic") for i in range(30)]
        assert len(_distinct_alternatives(others)) == 5
