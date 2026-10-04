"""The offline ranking benchmark: 62 hand-graded cases, no network, no randomness.

Design rules for this corpus, in priority order:

1. **Grades are about the dataset, not the metadata.** A dataset that publishes
   nothing but is exactly the right data is graded 3. A dataset with a beautiful
   metadata record about the wrong thing is graded 0 or -1. Otherwise the
   benchmark rewards loudness, which is the failure mode this whole evaluation
   exists to measure.
2. **Every candidate is graded.** No case is allowed to hide an ungraded
   distractor, because an ungraded distractor is a place to look good.
3. **Hard cases are the point.** Each case carries at least one of: sparse
   metadata on the right dataset, a synonym-only target, a popular irrelevant
   dataset, a metadata-inferred AI suggestion, or a deliberately missing answer.
4. **No fabricated registry references.** Candidate ids are
   ``benchmark:<case>/<key>`` and their source is ``kaggle-fixture``. Dataset
   *names* echo real datasets where the case is about a real one, because the
   name is part of what a user sees; the ids never pretend to resolve.

Grade legend: ``3`` highly relevant, ``2`` relevant, ``1`` weak,
``0`` irrelevant, ``-1`` incompatible. See ``labels.py`` for definitions.
"""

from __future__ import annotations

from app.schemas.dataset import SourceType
from app.services.datasets.evaluation.fixtures import BENCHMARK_VERSION
from app.services.datasets.evaluation.models import (
    BenchmarkCase,
    CandidateSpec,
    Difficulty,
    RequirementSpec,
)

# ---------------------------------------------------------------------------
# Publisher-signal buckets
#
# Repeated verbatim across cases so that popularity, recency and quality can be
# varied independently of relevance. Using named buckets instead of per-case
# numbers is what makes an ablation result interpretable: when NDCG moves, the
# only thing that moved is the signal under test.
# ---------------------------------------------------------------------------

HOT = dict(downloads=1_450_000, votes=5200, usability=0.95, updated_at="2026-09-08T00:00:00Z")
WARM = dict(downloads=210_000, votes=1050, usability=0.84, updated_at="2026-05-18T00:00:00Z")
MID = dict(downloads=38_000, votes=320, usability=0.72, updated_at="2025-11-12T00:00:00Z")
COLD = dict(downloads=4_200, votes=48, usability=0.62, updated_at="2024-02-09T00:00:00Z")
STALE = dict(downloads=9_800, votes=85, usability=0.55, updated_at="2021-06-30T00:00:00Z")
UNPUBLISHED = None  # candidate declares no popularity/recency/quality at all


class CaseBuilder:
    """Fluent builder so a case reads as a pool rather than as dict plumbing."""

    def __init__(
        self,
        *,
        id: str,
        description: str,
        category: str,
        requirements: dict,
        difficulty: Difficulty = Difficulty.STANDARD,
        retrieval_complete: bool = True,
        notes: str = "",
    ) -> None:
        self._case = BenchmarkCase(
            id=id,
            description=description,
            category=category,
            requirements=RequirementSpec(**requirements),
            candidates=[],
            grades={},
            difficulty=difficulty,
            retrieval_complete=retrieval_complete,
            notes=notes,
        )

    def add(
        self,
        grade: int,
        key: str,
        name: str,
        *,
        pub: dict | None = WARM,
        ai: bool = False,
        **spec,
    ) -> "CaseBuilder":
        """Add one graded candidate to the pool."""
        payload: dict = {"key": key, "name": name}
        if pub:
            payload.update(pub)
        if ai:
            payload["source_type"] = SourceType.AI_SUGGESTED
            payload["possible_source"] = spec.pop("possible_source", "Unknown")
        payload.update(spec)
        self._case.candidates.append(CandidateSpec(**payload))
        self._case.grades[key] = grade
        return self

    def ai_rationale(self, key: str, text: str) -> "CaseBuilder":
        """Attach tempting AI prose to a candidate, to prove it is not matched."""
        for spec in self._case.candidates:
            if spec.key == key:
                spec.rationale = text
        return self

    def build(self) -> BenchmarkCase:
        return self._case


def case(**kwargs) -> CaseBuilder:
    return CaseBuilder(**kwargs)


# ---------------------------------------------------------------------------
# EDUCATION
# ---------------------------------------------------------------------------

_EDU_PERFORMANCE = dict(
    domain="Education",
    subdomain="Educational Analytics",
    tasks=["classification", "regression"],
    target="academic performance",
    features=["attendance", "previous examination marks", "assignment scores", "study hours"],
    keywords=["student performance", "attendance", "examination", "assignment scores", "study hours"],
    dictionary_keywords=["student performance prediction"],
    data_types=["tabular data"],
)

EDUCATION: list[BenchmarkCase] = [
    case(
        id="edu-student-performance",
        description="Predict university students' final academic performance from attendance, prior marks and study habits.",
        category="education",
        requirements=_EDU_PERFORMANCE,
        notes="Baseline case: rich metadata on the right dataset, plus generic lookalikes.",
    )
    .add(
        3,
        "full-cohort",
        "Higher Education Students Academic Performance",
        domain="Education",
        tags=["education", "students", "university", "grades"],
        description="Records of university students with attendance, previous examination marks, assignment scores and study hours, labelled with final academic performance.",
        target_description="final academic performance grade",
        feature_description="columns include attendance, previous examination marks, assignment scores and study hours",
        task_types=["classification", "regression"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        2,
        "marks-only",
        "Student Grades Archive",
        domain="Education",
        tags=["education", "grades", "students"],
        description="Historical student grades and exam results by course and semester.",
        target_description="final grade",
        feature_description="columns include course, semester and grade",
        task_types=["regression"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "enrollment-stats",
        "University Enrollment Statistics",
        domain="Education",
        tags=["education", "enrollment", "statistics"],
        description="Aggregate university enrolment counts by faculty, country and year.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        1,
        "exam-archive-cold",
        "Examination Results Archive 2004-2018",
        domain="Education",
        tags=["education", "examination", "results"],
        description="Archived examination result sheets scanned from paper records.",
        file_types=["pdf", "txt"],
        pub=COLD,
    )
    .add(
        2,
        "ai-habits",
        "Student Habits Survey Study",
        description="Survey responses on study habits and attendance for secondary school students.",
        target_description="final grade",
        feature_description="attendance and study hours",
        task_types=["classification", "regression"],
        file_types=["csv"],
        ai=True,
        possible_source="Zenodo",
        pub=UNPUBLISHED,
    )
    .add(
        0,
        "world-rankings",
        "World University Rankings 2026",
        domain="Education",
        tags=["education", "rankings", "universities"],
        description="Composite global university rankings with scores per institution.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="edu-dropout-risk",
        description="Predict whether a student drops out before graduation; the right dataset publishes almost no metadata.",
        category="education",
        requirements=dict(
            domain="Education",
            subdomain="Educational Analytics",
            tasks=["classification"],
            target="dropout",
            features=["enrolled credits", "absences", "previous grades"],
            keywords=["student dropout", "enrolled credits", "absences", "retention"],
            dictionary_keywords=["student performance prediction"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SPARSE_METADATA,
        notes="Correct dataset has no feature/target descriptions, so the ranker must not treat silence as absence.",
    )
    .add(
        3,
        "sparse-dropout",
        "University Student Enrolment and Progression Records",
        domain="Education",
        tags=["education", "students", "enrolment"],
        description="Longitudinal records of student enrolment, credit attempts and progression outcomes.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        2,
        "declared-dropout",
        "Higher Education Dropout Dataset",
        domain="Education",
        tags=["education", "dropout", "students"],
        description="Higher education records with a published dropout indicator.",
        target_description="dropout status",
        feature_description="columns include enrolled credits, absences and previous grades",
        task_types=["classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "primary-enrolment",
        "Primary School Enrolment by District",
        domain="Education",
        tags=["education", "enrolment"],
        description="District-level primary school enrolment counts.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="edu-final-grade-semantic",
        description="Target phrased as 'final grade'; the candidate publishes the same label under an institutional name.",
        category="education",
        requirements=dict(
            domain="Education",
            subdomain="Educational Analytics",
            tasks=["classification", "regression"],
            target="final grade",
            features=["G3", "G2", "attendance"],
            keywords=["final grade", "G3", "attendance", "secondary school"],
            dictionary_keywords=["student performance prediction"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SEMANTIC,
        notes="Equivalence must be judged on the label space, not on the string 'final grade'.",
    )
    .add(
        3,
        "portuguese-school",
        "Portuguese Secondary School Student Performance",
        domain="Education",
        tags=["education", "students", "grades"],
        description="Secondary school student records with final grade G3, grade point average G2 and absence records.",
        target_description="G3 final grade",
        feature_description="columns include G3, G2, absences and study time",
        task_types=["classification", "regression"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "gpa-collections",
        "GPA Collections by University",
        domain="Education",
        tags=["education", "gpa"],
        description="Collection of grade point averages reported per university.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="edu-attendance-timeseries",
        description="Predict daily attendance rate per course across a semester from dated register entries.",
        category="education",
        requirements=dict(
            domain="Education",
            subdomain="Learning Analytics",
            tasks=["forecasting"],
            target="daily attendance rate",
            features=["attendance date", "course id", "student id"],
            keywords=["attendance", "daily", "course", "semester", "register"],
            dictionary_keywords=["student performance prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "daily-register",
        "Course Attendance Registers by Date",
        domain="Education",
        tags=["education", "attendance", "register"],
        description="Dated register entries per course and student across semesters.",
        target_description="daily attendance rate",
        feature_description="columns include attendance date, course id and student id",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        1,
        "monthly-aggregate",
        "Monthly Course Attendance Summary",
        domain="Education",
        tags=["education", "attendance"],
        description="Monthly aggregate attendance counts per course with no student-level rows.",
        target_description="monthly attendance rate",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "building-occupancy",
        "Campus Building Occupancy Counts",
        domain="Education",
        tags=["education", "campus"],
        description="Hourly head counts in campus buildings.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="edu-course-recommendation",
        description="Recommend the next course for a student from enrolment and rating interactions.",
        category="education",
        requirements=dict(
            domain="Education",
            subdomain="EdTech",
            tasks=["recommendation"],
            target="next recommended course",
            features=["student id", "course id", "enrollment", "course rating"],
            keywords=["course recommendation", "student", "enrollment", "rating"],
            dictionary_keywords=["product recommendation"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "mooc-enrollments",
        "MOOC Course Enrolment and Rating Interactions",
        domain="Education",
        tags=["education", "mooc", "courses"],
        description="Student-course enrolment matrix with per-course ratings and completion status.",
        target_description="course enrolment outcome",
        feature_description="columns include student id, course id, enrolment flag and course rating",
        task_types=["recommendation"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        1,
        "clicks-only",
        "Learning Management System Clickstream",
        domain="Education",
        tags=["education", "clickstream"],
        description="Clickstream events from a learning management system with no enrolment labels.",
        task_types=[],
        file_types=["json", "csv"],
        pub=WARM,
    )
    .add(
        0,
        "library-books",
        "Library Book Loans Archive",
        domain="Education",
        tags=["education", "library"],
        description="Book loan history from public libraries.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="edu-plagiarism-detection",
        description="Flag plagiarised submissions from submitted text, with a required label of plagiarism.",
        category="education",
        requirements=dict(
            domain="Education",
            subdomain="Assessment",
            tasks=["classification"],
            target="plagiarism",
            features=["submission text", "token overlap"],
            keywords=["plagiarism", "submission", "essay", "token overlap"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "essay-plagiarism",
        "Essay Corpus with Plagiarism Labels",
        domain="Education",
        tags=["education", "essays", "plagiarism"],
        description="Student essays annotated for plagiarised passages with matched source spans.",
        target_description="plagiarism flag",
        feature_description="submission text plus matched source token overlap",
        task_types=["classification"],
        file_types=["csv", "txt"],
        pub=MID,
    )
    .add(
        0,
        "essay-topics",
        "Essay Topic Labelling Benchmark",
        domain="Education",
        tags=["education", "essays"],
        description="Essays labelled by topic for text classification research.",
        target_description="essay topic label",
        feature_description="essay text",
        task_types=["text classification"],
        file_types=["txt"],
        pub=MID,
    )
    .add(
        -1,
        "handwritten-digits",
        "Handwritten Digit Image Collection",
        domain="Education",
        tags=["images", "digits"],
        description="Scanned images of handwritten digits with class labels.",
        target_description="digit class",
        task_types=["image classification"],
        file_types=["png"],
        pub=HOT,
    )
    .build(),
    case(
        id="edu-library-demand",
        description="Forecast future borrow counts per book title from checkout history.",
        category="education",
        requirements=dict(
            domain="Education",
            subdomain="Libraries",
            tasks=["forecasting"],
            target="borrow count",
            features=["checkout history", "book title", "member id"],
            keywords=["library", "borrow", "checkout", "book", "demand"],
            dictionary_keywords=["energy demand forecasting"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "library-loans",
        "Public Library Loan History",
        domain="Education",
        tags=["libraries", "loans"],
        description="Dated checkout history for public library titles.",
        target_description="number of borrows per title",
        feature_description="checkout history with book title and member id",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        1,
        "bookstore-sales",
        "Independent Bookstore Sales",
        domain="Retail",
        tags=["retail", "books"],
        description="Weekly unit sales of books in independent bookstores.",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="edu-generic-term-trap",
        description="Project description is dominated by generic education words; a popular aggregate dataset must not win.",
        category="education",
        requirements=dict(
            domain="Education",
            subdomain="Assessment Analytics",
            tasks=["regression"],
            target="course withdrawal rate",
            features=["withdrawal deadline", "course load", "advising contact"],
            keywords=["withdrawal", "course load", "advising", "students"],
            dictionary_keywords=["student performance prediction"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.DISTRACTOR,
        notes="Generic vocabulary (students, grades, performance) must not outweigh project-specific signals.",
    )
    .add(
        3,
        "withdrawal-microdata",
        "Course Withdrawal Microdata for Advising",
        domain="Education",
        tags=["education", "withdrawal"],
        description="Student-level records with withdrawal deadlines, course load and advising contact outcomes.",
        target_description="course withdrawal rate",
        feature_description="columns include withdrawal deadline, course load and advising contact",
        task_types=["classification", "regression"],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        0,
        "student-performance-generic",
        "Student Performance Indicators (Aggregated)",
        domain="Education",
        tags=["education", "students", "performance"],
        description="Aggregated student performance indicators across many institutions.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
]


# ---------------------------------------------------------------------------
# HEALTHCARE
# ---------------------------------------------------------------------------

HEALTHCARE: list[BenchmarkCase] = [
    case(
        id="health-diabetes-bmi",
        description="Predict diabetes from BMI, glucose and insulin measurements.",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Clinical Analysis",
            tasks=["classification"],
            target="diabetes",
            features=["bmi", "blood glucose", "insulin", "age"],
            keywords=["diabetes", "bmi", "blood glucose", "insulin"],
            dictionary_keywords=["chronic disease", "disease prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "pima-diabetes",
        "Pima Indian Diabetes Dataset",
        domain="Healthcare",
        tags=["health", "diabetes"],
        description="Female patients of Pima Indian heritage with BMI, blood glucose and insulin levels and diabetes outcome.",
        target_description="diabetes outcome",
        feature_description="columns include BMI, blood glucose, insulin and age",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "framingham",
        "Framingham Heart Study Cohort",
        domain="Healthcare",
        tags=["health", "cardiovascular"],
        description="Longitudinal cardiovascular cohort with blood glucose and cholesterol measures.",
        target_description="diabetes status",
        feature_description="columns include glucose, cholesterol, age and BMI",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "insurance",
        "Medical Insurance Cost Dataset",
        domain="Finance",
        tags=["insurance", "health"],
        description="Insurance charges and demographic factors for health policy holders.",
        task_types=["regression"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        0,
        "drug-reviews",
        "Drug Review Sentiment Dataset",
        domain="Healthcare",
        tags=["health", "reviews"],
        description="Patient drug reviews with effectiveness ratings.",
        target_description="review sentiment",
        task_types=["sentiment analysis"],
        file_types=["csv", "txt"],
        pub=HOT,
    )
    .build(),
    case(
        id="health-heart-disease",
        description="Predict presence of heart disease from age, cholesterol and blood pressure.",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Clinical Analysis",
            tasks=["classification"],
            target="heart disease",
            features=["age", "cholesterol", "resting blood pressure", "maximum heart rate"],
            keywords=["heart disease", "cholesterol", "blood pressure", "age"],
            dictionary_keywords=["disease prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "cleveland-heart",
        "Heart Disease Dataset (Cleveland)",
        domain="Healthcare",
        tags=["health", "heart", "cardiology"],
        description="Cardiology patients with age, cholesterol, resting blood pressure and maximum heart rate, labelled for heart disease.",
        target_description="heart disease presence",
        feature_description="columns include age, cholesterol, resting blood pressure and maximum heart rate",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "student-health",
        "Student Health and Lifestyle Survey",
        domain="Healthcare",
        tags=["health", "students"],
        description="Self-reported health and lifestyle habits of university students.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        3,
        "ai-heart-disease",
        "Cardiology Cohort Study (unlisted)",
        description="Cardiology cohort with age, cholesterol, resting blood pressure and maximum heart rate.",
        target_description="heart disease presence",
        feature_description="columns include age, cholesterol, resting blood pressure and maximum heart rate",
        task_types=["classification"],
        file_types=["csv"],
        ai=True,
        possible_source="PhysioNet",
        pub=UNPUBLISHED,
    )
    .ai_rationale(
        "ai-heart-disease",
        "This cohort is labelled for heart disease presence and carries exactly the "
        "cholesterol and blood pressure columns the project requires.",
    )
    .build(),
    case(
        id="health-readmission-sparse",
        description="Predict 30-day hospital readmission; the correct dataset publishes no features or target.",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Clinical Analysis",
            tasks=["classification"],
            target="readmission within 30 days",
            features=["discharge disposition", "encounter count", "diagnosis count"],
            keywords=["readmission", "discharge", "encounter", "hospital"],
            dictionary_keywords=["hospital readmission prediction"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SPARSE_METADATA,
        notes="Silence about target and features is UNKNOWN, never MISMATCH.",
    )
    .add(
        3,
        "sparse-ehr",
        "Hospital Encounters Longitudinal Extract",
        domain="Healthcare",
        tags=["health", "hospital"],
        description="Anonymised hospital encounter extracts with discharge and encounter identifiers.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        1,
        "generic-readmission-hints",
        "Care Planning Notes Collection",
        domain="Healthcare",
        tags=["health", "notes"],
        description="Free-text care planning notes with limited structured fields.",
        task_types=["text classification"],
        file_types=["txt", "csv"],
        pub=COLD,
    )
    .build(),
    case(
        id="health-retinal-imaging",
        description="Detect diabetic retinopathy from retinal fundus photographs.",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Medical Imaging",
            tasks=["image classification"],
            target="diabetic retinopathy grade",
            features=["fundus photograph"],
            keywords=["retinal", "fundus", "diabetic retinopathy", "screening"],
            dictionary_keywords=["retinal image analysis", "medical image diagnosis"],
            data_types=["image data"],
        ),
    )
    .add(
        3,
        "aptos-blindness",
        "Diabetic Retinopathy Blindness Detection",
        domain="Healthcare",
        tags=["health", "retina", "imaging"],
        description="Retinal fundus photographs labelled by diabetic retinopathy severity grade.",
        target_description="diabetic retinopathy grade",
        feature_description="fundus photograph per eye",
        task_types=["image classification"],
        file_types=["png", "jpg"],
        pub=HOT,
    )
    .add(
        1,
        "tabular-diabetes",
        "Diabetes Clinical Indicators (Tabular)",
        domain="Healthcare",
        tags=["health", "diabetes"],
        description="Tabular clinical indicators for diabetes risk with no imaging.",
        target_description="diabetes risk level",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "heart-sounds",
        "Heart Sound Recordings",
        domain="Healthcare",
        tags=["health", "audio"],
        description="Audio recordings of heart sounds for murmur classification.",
        target_description="murmur flag",
        task_types=["audio classification"],
        file_types=["wav"],
        pub=WARM,
    )
    .add(
        -1,
        "tabular-eye-disease",
        "Eye Disease Diagnosis Records (Tabular Only)",
        domain="Healthcare",
        tags=["health", "eyes"],
        description="Tabular clinical records of eye disease diagnoses with no fundus imagery.",
        target_description="eye disease diagnosis",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="health-target-conflict",
        description="Project needs mortality; the candidate positively declares a different label (readmission).",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Clinical Analysis",
            tasks=["classification"],
            target="mortality",
            features=["age", "admission type", "diagnosis"],
            keywords=["mortality", "death", "admission", "diagnosis"],
            dictionary_keywords=["disease prediction"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.DISTRACTOR,
        notes="Declared label conflict is real incompatibility evidence, unlike silence.",
    )
    .add(
        -1,
        "declared-readmission",
        "Inpatient Readmission Study",
        domain="Healthcare",
        tags=["health", "hospital"],
        description="Inpatient records with a declared readmission outcome.",
        target_description="readmission outcome",
        feature_description="columns include age, admission type and diagnosis",
        task_types=["classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        1,
        "sparse-outcomes",
        "Hospital Outcome Notes (Unlabelled)",
        domain="Healthcare",
        tags=["health", "hospital"],
        description="Unlabelled hospital outcome notes mentioning cases and outcomes.",
        task_types=[],
        file_types=["txt"],
        pub=COLD,
    )
    .build(),
    case(
        id="health-genomics-response",
        description="Predict treatment response from gene expression; raw assay data needs substantial preprocessing.",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Genomics",
            tasks=["classification"],
            target="treatment response",
            features=["gene expression", "dose", "cell line"],
            keywords=["gene expression", "treatment response", "dose", "cell line"],
            dictionary_keywords=["gene expression analysis"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SEMANTIC,
    )
    .add(
        3,
        "response-labeled",
        "Drug Response with Labelled Outcomes",
        domain="Healthcare",
        tags=["health", "genomics"],
        description="Gene expression profiles paired with measured treatment response labels.",
        target_description="treatment response class",
        feature_description="gene expression matrix with dose and cell line identifiers",
        task_types=["classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        1,
        "raw-assay",
        "Raw Gene Expression Assay Dump",
        domain="Healthcare",
        tags=["health", "genomics"],
        description="Raw gene expression assay dump with no treatment outcome labels.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .build(),
    case(
        id="health-adverse-effect-nlp",
        description="Classify adverse drug effects from patient review text.",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Pharmacovigilance",
            tasks=["text classification"],
            target="adverse effect",
            features=["review text", "drug name"],
            keywords=["adverse effect", "drug", "review", "side effect"],
            dictionary_keywords=["disease prediction"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "ade-corpus",
        "Adverse Drug Event Annotation Corpus",
        domain="Healthcare",
        tags=["health", "drugs", "nlp"],
        description="Patient drug review text annotated with adverse drug event relations.",
        target_description="adverse drug event relation",
        feature_description="review text with drug mention spans",
        task_types=["text classification"],
        file_types=["txt", "csv"],
        pub=MID,
    )
    .add(
        2,
        "drug-review-sentiment",
        "Drug Review Sentiment Dataset",
        domain="Healthcare",
        tags=["health", "reviews"],
        description="Patient drug reviews with usefulness ratings.",
        target_description="review sentiment",
        feature_description="review text",
        task_types=["sentiment analysis"],
        file_types=["csv", "txt"],
        pub=HOT,
    )
    .add(
        -1,
        "movie-reviews",
        "Large Movie Review Dataset",
        domain="Entertainment",
        tags=["movies", "reviews"],
        description="IMDB movie review polarity data.",
        target_description="sentiment label",
        feature_description="review text",
        task_types=["sentiment analysis"],
        file_types=["txt"],
        pub=HOT,
    )
    .build(),
    case(
        id="health-age-bias-audit",
        description="Audit age bias in an outcome predictor; age is the protected attribute and outcome the target.",
        category="healthcare",
        requirements=dict(
            domain="Healthcare",
            subdomain="Fairness",
            tasks=["classification", "clustering"],
            target="treatment outcome",
            features=["age", "sex", "comorbidity count", "length of stay"],
            keywords=["age", "bias", "fairness", "outcome", "protected attribute"],
            dictionary_keywords=["disease prediction"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SEMANTIC,
    )
    .add(
        3,
        "fairness-benchmark",
        "Clinical Outcome Fairness Benchmarks",
        domain="Healthcare",
        tags=["health", "fairness"],
        description="Clinical outcome records annotated with protected attributes for bias auditing.",
        target_description="treatment outcome class",
        feature_description="columns include age, sex, comorbidity count and length of stay",
        task_types=["classification"],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        0,
        "credit-bias",
        "Credit Scoring Records for Bias Review",
        domain="Finance",
        tags=["finance", "bias"],
        description="Credit scoring records with age and approval outcomes.",
        target_description="approval outcome",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
]


# ---------------------------------------------------------------------------
# CYBERSECURITY
# ---------------------------------------------------------------------------

CYBERSECURITY: list[BenchmarkCase] = [
    case(
        id="cyber-iot-intrusion",
        description="Classify intrusion type in IoT network traffic from flow statistics.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Network Security",
            tasks=["classification", "anomaly detection"],
            target="intrusion type",
            features=["source ip", "destination ip", "protocol", "byte count"],
            keywords=["intrusion", "iot", "network traffic", "protocol", "byte count"],
            dictionary_keywords=["intrusion detection", "network traffic analysis"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "bot-iot",
        "BoT-IoT Dataset",
        domain="Cybersecurity",
        tags=["security", "iot", "network"],
        description="IoT and IIoT network traffic labelled by botnet, DDoS and intrusion type with flow statistics.",
        target_description="intrusion type class",
        feature_description="columns include source ip, destination ip, protocol and byte count",
        task_types=["classification", "anomaly detection"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "nba-iot",
        "N-BaIoT Device Anomaly Dataset",
        domain="Cybersecurity",
        tags=["security", "iot", "anomaly"],
        description="Commercial IoT device traffic captured before and during infection, labelled for anomaly.",
        target_description="anomaly class",
        feature_description="device traffic flow features per device",
        task_types=["classification", "anomaly detection"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "smart-home-energy",
        "Smart Home Energy Consumption Dataset",
        domain="IoT",
        tags=["iot", "energy", "smart home"],
        description="Smart home appliance energy consumption traces.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        -1,
        "fifa-gameplay",
        "FIFA Video Game Gameplay Network Data",
        domain="Entertainment",
        tags=["gaming", "network"],
        description="Network traffic traces captured from a video game session.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="cyber-phishing-url",
        description="Detect malicious URLs from raw URL strings.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Web Security",
            tasks=["classification"],
            target="phishing",
            features=["url string", "url length", "num underscores"],
            keywords=["phishing", "malicious url", "url", "website"],
            dictionary_keywords=["phishing"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "phi-url",
        "Phishing Website URLs Dataset",
        domain="Cybersecurity",
        tags=["security", "urls", "phishing"],
        description="Labelled URLs of phishing and benign websites with lexical URL features.",
        target_description="phishing label",
        feature_description="url string with url length and num underscores counts",
        task_types=["classification"],
        file_types=["csv", "txt"],
        pub=WARM,
    )
    .add(
        1,
        "malware-urls",
        "Malware and Benign URL Strings",
        domain="Cybersecurity",
        tags=["security", "urls", "malware"],
        description="URL strings labelled for malware hosting rather than phishing.",
        target_description="malware label",
        feature_description="url string tokens",
        task_types=["classification"],
        file_types=["txt"],
        pub=MID,
    )
    .add(
        0,
        "adult-websites",
        "Adult Website Classification Dataset",
        domain="Web",
        tags=["web", "urls"],
        description="URL and content classification for adult websites.",
        target_description="adult content label",
        task_types=["classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="cyber-adversarial-weak",
        description="Adversarial detection needs attack generation; no public dataset ships attacks, so grade 1.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Network Security",
            tasks=["anomaly detection"],
            target="adversarial attack",
            features=["adversarial perturbation", "clean input"],
            keywords=["adversarial", "perturbation", "attack generation"],
            dictionary_keywords=["intrusion detection", "malware detection"],
            data_types=["image data"],
        ),
        difficulty=Difficulty.SEMANTIC,
        notes="Requires synthesising attacks: WEAK by definition, not MISSING.",
    )
    .add(
        1,
        "mnist-defense",
        "MNIST with Adversarial Samples (Small Split)",
        domain="Artificial Intelligence",
        tags=["digits", "adversarial"],
        description="A small MNIST-derived split with a handful of adversarial samples.",
        target_description="adversarial flag",
        task_types=["classification"],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        0,
        "fashion-mnist",
        "Fashion-MNIST Image Collection",
        domain="Artificial Intelligence",
        tags=["images", "fashion"],
        description="Grayscale images of clothing categories.",
        task_types=["image classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="cyber-rare-attack-subtype",
        description="Need a specific attack type that is only a subset of a broader intrusion dataset.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Network Security",
            tasks=["classification"],
            target="specific attack type",
            features=["packet length", "flag count", "protocol type"],
            keywords=["attack type", "packet", "protocol", "flag"],
            dictionary_keywords=["intrusion detection"],
            data_types=["tabular data"],
        ),
    )
    .add(
        2,
        "cicids-subset",
        "Network Intrusion Flow Records (Broad Taxonomy)",
        domain="Cybersecurity",
        tags=["security", "network"],
        description="Flow records across a broad intrusion taxonomy; the required attack is one of many classes.",
        target_description="attack type taxonomy label",
        feature_description="columns include packet length, flag count and protocol type",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        -1,
        "weather-stations",
        "Weather Station Telemetry Archive",
        domain="Environment",
        tags=["weather", "telemetry"],
        description="Telemetry from weather stations measuring pressure and wind.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .build(),
    case(
        id="cyber-code-vulnerability",
        description="Detect source-code vulnerabilities from code snippets.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Application Security",
            tasks=["classification", "detection"],
            target="vulnerability presence",
            features=["source code", "language", "cwe id"],
            keywords=["vulnerability", "source code", "cwe", "static analysis"],
            dictionary_keywords=["malware detection"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "nvd-cwe",
        "National Vulnerability Database Extract",
        domain="Cybersecurity",
        tags=["security", "vulnerabilities"],
        description="Source code snippets annotated with CWE vulnerability identifiers.",
        target_description="vulnerability presence label",
        feature_description="source code snippet with language and CWE id",
        task_types=["classification", "detection"],
        file_types=["csv", "txt"],
        pub=WARM,
    )
    .add(
        2,
        "cve-mentions",
        "CVE Mention Corpus",
        domain="Cybersecurity",
        tags=["security", "cve"],
        description="Text mentions of CVEs from advisories, without code-level labels.",
        target_description="vulnerability mention",
        feature_description="advisory text",
        task_types=["text classification"],
        file_types=["txt"],
        pub=MID,
    )
    .add(
        0,
        "excel-vuln-reports",
        "Excel Vulnerability Reports Archive",
        domain="Cybersecurity",
        tags=["security", "reports"],
        description="Archived spreadsheet vulnerability reports.",
        task_types=[],
        file_types=["csv"],
        pub=STALE,
    )
    .build(),
    case(
        id="cyber-sms-spam",
        description="Classify SMS messages as spam or ham.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Spam Filtering",
            tasks=["classification", "text classification"],
            target="spam",
            features=["message text", "sender"],
            keywords=["sms", "spam", "ham", "message"],
            dictionary_keywords=["spam detection"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "sms-spam",
        "SMS Spam Collection (UCI)",
        domain="Cybersecurity",
        tags=["sms", "spam"],
        description="SMS messages labelled spam or ham with sender metadata.",
        target_description="spam label",
        feature_description="message text and sender",
        task_types=["classification"],
        file_types=["txt", "csv"],
        pub=WARM,
    )
    .add(
        2,
        "email-spam",
        "Email Spam Enron Corpus",
        domain="Cybersecurity",
        tags=["email", "spam"],
        description="Email corpus labelled spam or ham (longer messages than SMS).",
        target_description="spam label",
        feature_description="email body text",
        task_types=["text classification"],
        file_types=["txt"],
        pub=WARM,
    )
    .add(
        1,
        "fake-news-text",
        "Fake News Article Corpus",
        domain="Social Media",
        tags=["news", "text"],
        description="News articles labelled real or fake (not spam).",
        target_description="fake news label",
        feature_description="article text",
        task_types=["text classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="cyber-audio-command",
        description="Detect spoken voice commands from audio in a security system.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Audio Security",
            tasks=["classification"],
            target="spoken command",
            features=["audio waveform", "speaker id"],
            keywords=["audio", "voice", "command", "speaker"],
            dictionary_keywords=["intrusion detection"],
            data_types=["audio data"],
        ),
    )
    .add(
        3,
        "audio-intrusion",
        "Audio-Triggered Intrusion Detection Dataset",
        domain="Cybersecurity",
        tags=["security", "audio"],
        description="Audio recordings of spoken security commands labelled by command class.",
        target_description="spoken command class",
        feature_description="audio waveform and speaker id",
        task_types=["classification"],
        file_types=["wav"],
        pub=COLD,
    )
    .add(
        0,
        "tabular-intrusion",
        "Tabular Network Intrusion Features",
        domain="Cybersecurity",
        tags=["security", "network"],
        description="Tabular intrusion features with no audio.",
        target_description="intrusion class",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="cyber-ai-surfaced-dataset",
        description="The strongest dataset is an unverified AI suggestion; verification weight must not bury it silently.",
        category="cybersecurity",
        requirements=dict(
            domain="Cybersecurity",
            subdomain="Malware Analysis",
            tasks=["classification"],
            target="malware family",
            features=["api call sequence", "file hash"],
            keywords=["malware", "api calls", "file hash", "family"],
            dictionary_keywords=["malware detection"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SEMANTIC,
        notes="Verification (weight 15) must not invert a clear relevance win.",
    )
    .add(
        3,
        "ai-malware",
        "Malware API Call Sequence Corpus",
        description="API call sequences with file hashes labelled by malware family.",
        target_description="malware family class",
        feature_description="api call sequence and file hash per sample",
        task_types=["classification"],
        file_types=["csv"],
        ai=True,
        possible_source="MalwareBazaar",
        pub=UNPUBLISHED,
    )
    .ai_rationale(
        "ai-malware",
        "This corpus contains the api call sequence and file hash columns and is the canonical malware family benchmark.",
    )
    .add(
        0,
        "verified-unrelated",
        "Verified Endpoint Telemetry Summary",
        domain="Cybersecurity",
        tags=["security", "telemetry"],
        description="Verified endpoint telemetry summaries with device inventory counts.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
]


# ---------------------------------------------------------------------------
# FINANCE
# ---------------------------------------------------------------------------

FINANCE: list[BenchmarkCase] = [
    case(
        id="finance-churn",
        description="Predict bank customer churn (Exited = 1/0) from balance, tenure and product holdings.",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Customer Analytics",
            tasks=["classification"],
            target="Exited",
            features=["balance", "tenure", "num products", "estimated salary"],
            keywords=["customer churn", "exited", "balance", "tenure", "bank"],
            dictionary_keywords=["customer churn prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "churn-banking",
        "Bank Customer Churn (Exited label)",
        domain="Finance",
        tags=["bank", "churn"],
        description="Bank customers with balance, tenure and product holdings; the Exited column marks churn (1/0).",
        target_description="Exited (1 = customer churned, 0 = retained)",
        feature_description="columns include balance, tenure, num products and estimated salary",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        3,
        "churn-telecom",
        "Telecom Customer Churn",
        domain="Telecommunications",
        tags=["telecom", "churn"],
        description="Telecom churn dataset with contract, tenure and monthly charges.",
        target_description="Churn (Yes/No)",
        feature_description="columns include contract, tenure and monthly charges",
        task_types=["classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        1,
        "churn-retail",
        "Retail Loyalty Churn Survey",
        domain="Retail",
        tags=["retail", "churn"],
        description="Retail loyalty survey responses about churn intent (needs recoding).",
        target_description="churn intent score",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "generic-customer",
        "Customer Segmentation Profiles",
        domain="Retail",
        tags=["customers", "segmentation"],
        description="Customer profiles without a churn outcome.",
        task_types=["clustering"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="finance-credit-default",
        description="Predict loan default from income, credit amount and duration.",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Risk",
            tasks=["classification"],
            target="default",
            features=["income", "credit amount", "duration", "savings account"],
            keywords=["credit default", "loan", "income", "risk"],
            dictionary_keywords=["credit scoring"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "german-credit",
        "German Credit Risk Dataset",
        domain="Finance",
        tags=["credit", "risk"],
        description="Loan applicants with income, credit amount, duration and savings, labelled for default risk.",
        target_description="default risk class",
        feature_description="columns include credit amount, duration, savings and instalment plans",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "loan-playground",
        "Loan Approval Prediction Playground",
        domain="Finance",
        tags=["loans", "classification"],
        description="Loan application records labelled for approval status (not default).",
        target_description="loan status (Approved/Rejected)",
        feature_description="columns include income, loan amount and term",
        task_types=["classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        1,
        "credit-card-tx",
        "Credit Card Transaction Records",
        domain="Finance",
        tags=["credit card", "transactions"],
        description="Anonymised credit card transactions without default labels.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="finance-fraud",
        description="Detect fraudulent card transactions (isFraud target).",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Fraud",
            tasks=["classification"],
            target="isFraud",
            features=["transaction amount", "card brand", "merchant category"],
            keywords=["fraud", "transaction", "isFraud", "card"],
            dictionary_keywords=["financial fraud detection"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "ieee-cis",
        "IEEE-CIS Fraud Detection",
        domain="Finance",
        tags=["fraud", "card"],
        description="Anonymised card transactions with an isFraud target column.",
        target_description="isFraud (1 = fraudulent transaction)",
        feature_description="columns include transaction amount, card brand and merchant category",
        task_types=["classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        2,
        "creditcard-fraud",
        "Credit Card Fraud Detection (ULB)",
        domain="Finance",
        tags=["fraud", "card"],
        description="European credit card transactions labelled fraudulent.",
        target_description="fraud class",
        feature_description="transaction amount, time and distance from last transaction",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "bank-transactions",
        "Bank Transaction Fraud Annotations",
        domain="Finance",
        tags=["bank", "fraud"],
        description="Bank account transactions with fraud annotations.",
        target_description="fraud flag",
        feature_description="amount, account type and merchant",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "online-retail-sales",
        "Online Retail Sales Transactions",
        domain="Retail",
        tags=["retail", "sales"],
        description="Online retail order lines with no fraud labels.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="finance-stock-forecast",
        description="Forecast next-day closing price from historical prices and volume.",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Trading",
            tasks=["forecasting", "regression"],
            target="next day close",
            features=["open", "high", "low", "volume"],
            keywords=["stock", "close", "price", "volume", "forecast"],
            dictionary_keywords=["stock price prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "stock-prices",
        "Daily Stock Price History",
        domain="Finance",
        tags=["stocks", "prices"],
        description="Daily open/high/low/close and volume for listed equities.",
        target_description="next day close price",
        feature_description="columns include open, high, low, close and volume",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        2,
        "bitcoin-prices",
        "Bitcoin Hourly Price Series",
        domain="Finance",
        tags=["crypto", "prices"],
        description="Hourly Bitcoin price series.",
        target_description="next hour close price",
        feature_description="open, high, low, close and volume",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        1,
        "retail-sales-ts",
        "Monthly Retail Sales Series",
        domain="Retail",
        tags=["retail", "sales"],
        description="Monthly aggregate retail sales (no per-asset prices).",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="finance-insurance-fraud",
        description="Detect fraudulent insurance claims from claim and policy records.",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Insurance",
            tasks=["classification"],
            target="claim fraud",
            features=["claim amount", "policy type", "prior claims"],
            keywords=["insurance", "fraud", "claim", "policy"],
            dictionary_keywords=["insurance fraud detection"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "insurance-claims-fraud",
        "Insurance Claims Fraud Dataset",
        domain="Finance",
        tags=["insurance", "fraud"],
        description="Insurance claims with amount, policy type and prior claims, labelled fraudulent.",
        target_description="fraudulent claim flag",
        feature_description="columns include claim amount, policy type and prior claims",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "auto-insurance",
        "Auto Insurance Claim Severity",
        domain="Finance",
        tags=["insurance", "auto"],
        description="Auto insurance claims labelled by severity rather than fraud.",
        target_description="claim severity class",
        feature_description="claim amount and vehicle type",
        task_types=["classification", "regression"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "medical-insurance",
        "Medical Insurance Charges",
        domain="Finance",
        tags=["insurance", "medical"],
        description="Medical insurance charge records without fraud labels.",
        task_types=["regression"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="finance-target-conflict-approved",
        description="Project needs whether a loan was approved; candidate positively declares a default label.",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Credit",
            tasks=["classification"],
            target="approved",
            features=["applicant income", "requested amount"],
            keywords=["loan", "approved", "applicant", "amount"],
            dictionary_keywords=["credit scoring"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.DISTRACTOR,
    )
    .add(
        -1,
        "declared-default",
        "Loan Default Registry",
        domain="Finance",
        tags=["loans", "default"],
        description="Loan applications labelled with a default outcome.",
        target_description="default outcome",
        feature_description="applicant income and requested amount",
        task_types=["classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        1,
        "sparse-applications",
        "Loan Application Notes (No Labels)",
        domain="Finance",
        tags=["loans", "notes"],
        description="Unlabelled loan application notes.",
        task_types=[],
        file_types=["txt"],
        pub=COLD,
    )
    .build(),
    case(
        id="finance-delinquent-semantic",
        description="Target phrased as 'delinquent'; the candidate uses 'charged off / past due'.",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Credit",
            tasks=["classification"],
            target="delinquent",
            features=["days past due", "balance"],
            keywords=["delinquent", "charged off", "past due", "balance"],
            dictionary_keywords=["credit scoring"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SEMANTIC,
        notes="Requires knowing delinquent ≈ charged off ≈ past due.",
    )
    .add(
        3,
        "chargeoff-semantic",
        "Charge-off and Delinquency Status Corpus",
        domain="Finance",
        tags=["credit", "delinquency"],
        description="Consumer credit accounts labelled charged off, delinquent or current by days past due.",
        target_description="delinquent status (charged off / past due label space)",
        feature_description="days past due and balance per account",
        task_types=["classification"],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        0,
        "credit-cards-general",
        "General Credit Card Accounts Summary",
        domain="Finance",
        tags=["credit card", "summary"],
        description="Aggregated credit card account summaries without delinquency labels.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="finance-ai-inferred-target",
        description="AI suggestion claims a 'revenue' target when the project needs profit; the rationale must not be matched.",
        category="finance",
        requirements=dict(
            domain="Finance",
            subdomain="Profitability",
            tasks=["regression", "forecasting"],
            target="net profit margin",
            features=["unit cost", "unit price", "overhead"],
            keywords=["profit margin", "unit cost", "unit price", "overhead"],
            dictionary_keywords=[],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SEMANTIC,
        notes="rationale contains 'net profit margin'; only declared fields may be matched.",
    )
    .add(
        3,
        "profit-margin",
        "Unit Economics with Net Profit Margin",
        domain="Finance",
        tags=["profitability", "unit economics"],
        description="Unit-level economics with unit cost, unit price, overhead and net profit margin.",
        target_description="net profit margin",
        feature_description="columns include unit cost, unit price and overhead",
        task_types=["regression"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "ai-revenue",
        "Retail Revenue Trend Report",
        description="Retail revenue trend report covering several quarters of order volume.",
        target_description="revenue trend",
        task_types=["forecasting"],
        file_types=["csv"],
        ai=True,
        possible_source="Internal analytics",
        pub=UNPUBLISHED,
    )
    .ai_rationale(
        "ai-revenue",
        "This dataset declares net profit margin alongside unit cost and unit price, making it the correct match.",
    )
    .build(),
]


# ---------------------------------------------------------------------------
# RETAIL
# ---------------------------------------------------------------------------

RETAIL: list[BenchmarkCase] = [
    case(
        id="retail-demand-forecast",
        description="Forecast weekly demand per store and SKU from price, promotion and calendar features.",
        category="retail",
        requirements=dict(
            domain="Retail",
            subdomain="Forecasting",
            tasks=["forecasting"],
            target="weekly demand",
            features=["store id", "sku", "price", "promotion flag", "week"],
            keywords=["demand", "sales", "store", "sku", "promotion"],
            dictionary_keywords=["sales forecasting"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "walmart-sales",
        "Walmart Recruiting II: Sales in Stormy Weather",
        domain="Retail",
        tags=["retail", "sales", "weekly"],
        description="Weekly store and SKU sales with price, promotion and weather features.",
        target_description="weekly unit sales (demand)",
        feature_description="columns include store id, sku, price, promo and week",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        2,
        "grocery-weekly",
        "Grocery Item Demand Weekly",
        domain="Retail",
        tags=["retail", "grocery"],
        description="Weekly grocery item demand counts with holiday flags.",
        target_description="weekly item demand",
        feature_description="date, item and demand count",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        1,
        "fashion-ecommerce",
        "Fashion Ecommerce Sales Snapshot",
        domain="Retail",
        tags=["fashion", "ecommerce"],
        description="Monthly fashion ecommerce sales snapshot without per-SKU demand.",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="retail-basket-analysis",
        description="Unsupervised market-basket analysis over grocery transactions (no target).",
        category="retail",
        requirements=dict(
            domain="Retail",
            subdomain="Basket Analysis",
            tasks=["clustering", "association rule mining"],
            target=None,
            features=["transaction id", "item id", "quantity"],
            keywords=["market basket", "grocery", "transaction", "item"],
            dictionary_keywords=["product recommendation"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "grocery-basket",
        "Online Grocery Transactions (Basket Ready)",
        domain="Retail",
        tags=["retail", "grocery"],
        description="Transaction-level grocery baskets ready for association rules.",
        feature_description="columns include transaction id, item id and quantity",
        task_types=["clustering", "association rule mining"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        1,
        "store-inventory",
        "Store Inventory Levels",
        domain="Retail",
        tags=["inventory"],
        description="Weekly store inventory levels per item.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        0,
        "music-listening",
        "Music Listening Sessions",
        domain="Entertainment",
        tags=["music", "sessions"],
        description="Music streaming listening sessions.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="retail-price-elasticity",
        description="Estimate units sold from price and promotion, needing both price and sales columns.",
        category="retail",
        requirements=dict(
            domain="Retail",
            subdomain="Pricing",
            tasks=["regression"],
            target="units sold",
            features=["price", "promotion flag", "week"],
            keywords=["units sold", "price", "promotion", "elasticity"],
            dictionary_keywords=["sales forecasting"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "price-sales",
        "Retail Price and Unit Sales Panel",
        domain="Retail",
        tags=["pricing", "sales"],
        description="Store-week panel with price, promotion flag and units sold.",
        target_description="units sold",
        feature_description="columns include price, promotion flag and units sold",
        task_types=["regression"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        1,
        "sales-only",
        "Aggregate Units Sold by Category",
        domain="Retail",
        tags=["sales"],
        description="Units sold per category per month with no price column.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="retail-segmentation",
        description="Label customer segments for targeted retail marketing.",
        category="retail",
        requirements=dict(
            domain="Retail",
            subdomain="Customer Analytics",
            tasks=["clustering", "classification"],
            target="customer segment",
            features=["recency", "frequency", "monetary"],
            keywords=["customer segment", "rfm", "marketing", "cluster"],
            dictionary_keywords=["customer churn prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "ecommerce-segment",
        "Ecommerce Customer Segmentation",
        domain="Retail",
        tags=["retail", "segmentation"],
        description="Ecommerce customers with recency, frequency, monetary and a published segment label.",
        target_description="customer segment label",
        feature_description="columns include recency, frequency and monetary value",
        task_types=["clustering", "classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "shoppers-generic",
        "Online Shoppers Purchasing Intention",
        domain="Retail",
        tags=["shoppers"],
        description="Online shopper sessions with purchase intention labels.",
        target_description="purchase intention (True/False)",
        task_types=["classification"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "gym-members",
        "Gym Membership Attendance",
        domain="Healthcare",
        tags=["fitness"],
        description="Gym member attendance records.",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="retail-review-rating",
        description="Predict the numeric product rating from review text.",
        category="retail",
        requirements=dict(
            domain="Retail",
            subdomain="Reviews",
            tasks=["regression", "sentiment analysis"],
            target="rating",
            features=["review text", "product id", "star rating"],
            keywords=["product review", "rating", "review text", "stars"],
            dictionary_keywords=["sentiment analysis"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "amazon-reviews",
        "Amazon Product Review Ratings",
        domain="Retail",
        tags=["retail", "reviews"],
        description="Amazon product reviews with 1-5 star ratings and product ids.",
        target_description="star rating",
        feature_description="review text with product id and star rating",
        task_types=["regression", "sentiment analysis"],
        file_types=["csv", "txt"],
        pub=HOT,
    )
    .add(
        2,
        "ratings-only",
        "Product Ratings Without Review Text",
        domain="Retail",
        tags=["ratings"],
        description="Product star ratings with no review body text.",
        target_description="star rating",
        task_types=["regression"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "movie-reviews-retail",
        "Movie Review Polarity",
        domain="Entertainment",
        tags=["movies"],
        description="IMDB review polarity labels (binary, not star ratings).",
        target_description="sentiment polarity",
        feature_description="review text",
        task_types=["sentiment analysis"],
        file_types=["txt"],
        pub=HOT,
    )
    .build(),
    case(
        id="retail-cart-abandonment",
        description="Predict abandoned shopping carts from session events.",
        category="retail",
        requirements=dict(
            domain="Retail",
            subdomain="Web Analytics",
            tasks=["classification", "forecasting"],
            target="cart abandonment",
            features=["session id", "cart value", "last event timestamp"],
            keywords=["cart abandonment", "session", "cart value"],
            dictionary_keywords=["product recommendation"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "cart-events",
        "Shopping Cart Session Events",
        domain="Retail",
        tags=["retail", "sessions"],
        description="Session-level cart events with abandonment flags and cart value.",
        target_description="cart abandonment flag",
        feature_description="columns include session id, cart value and last event timestamp",
        task_types=["classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        1,
        "web-analytics",
        "Generic Web Analytics Pageviews",
        domain="Web",
        tags=["web", "analytics"],
        description="Pageview analytics without cart semantics.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
]


# ---------------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------------

NLP: list[BenchmarkCase] = [
    case(
        id="nlp-multilingual-sentiment",
        description="Multilingual sentiment analysis across English, French and Arabic reviews.",
        category="nlp",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="NLP",
            tasks=["sentiment analysis", "text classification"],
            target="sentiment",
            features=["review text", "language", "star rating"],
            keywords=["sentiment", "multilingual", "reviews", "language"],
            dictionary_keywords=["sentiment analysis"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "multilingual-reviews",
        "Multilingual Product Reviews",
        domain="Retail",
        tags=["reviews", "multilingual"],
        description="Product reviews in three languages labelled positive or negative.",
        target_description="sentiment label",
        feature_description="review text with language and star rating",
        task_types=["sentiment analysis", "text classification"],
        file_types=["csv", "txt"],
        pub=WARM,
    )
    .add(
        2,
        "tweets",
        "Sentiment140 Tweet Corpus",
        domain="Social Media",
        tags=["tweets", "sentiment"],
        description="Tweets labelled positive or negative (English only, informal).",
        target_description="sentiment label",
        feature_description="tweet text",
        task_types=["sentiment analysis"],
        file_types=["txt", "csv"],
        pub=HOT,
    )
    .add(
        1,
        "one-language",
        "French Movie Review Corpus",
        domain="Entertainment",
        tags=["movies", "reviews"],
        description="French-only movie review sentiment.",
        target_description="sentiment label",
        feature_description="review text",
        task_types=["sentiment analysis"],
        file_types=["txt"],
        pub=WARM,
    )
    .add(
        0,
        "tabular-stars",
        "Structured Star Ratings Without Text",
        domain="Retail",
        tags=["ratings"],
        description="Star ratings with no review text.",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="nlp-banking-intent",
        description="Classify banking customer-service utterances into intents.",
        category="nlp",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="NLP",
            tasks=["text classification"],
            target="intent",
            features=["utterance text"],
            keywords=["intent", "banking", "utterance", "customer service"],
            dictionary_keywords=["text classification"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "bank77-intent",
        "Banking77 Intent Dataset",
        domain="Finance",
        tags=["banking", "intents"],
        description="Banking customer-service utterances labelled with one of 77 intents.",
        target_description="intent label",
        feature_description="utterance text",
        task_types=["text classification"],
        file_types=["csv", "txt"],
        pub=WARM,
    )
    .add(
        1,
        "sms-spam-intent",
        "SMS Spam Utterances",
        domain="Cybersecurity",
        tags=["sms", "spam"],
        description="SMS labelled spam/ham, not banking intent.",
        target_description="spam label",
        feature_description="message text",
        task_types=["text classification"],
        file_types=["txt"],
        pub=MID,
    )
    .add(
        0,
        "bbc-topics",
        "BBC News Topic Corpus",
        domain="Social Media",
        tags=["news", "topics"],
        description="News articles labelled by topic, not intent.",
        target_description="topic label",
        feature_description="article text",
        task_types=["topic modeling", "text classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="nlp-clinical-ner",
        description="Extract clinical entities from discharge summaries.",
        category="nlp",
        requirements=dict(
            domain="Healthcare",
            subdomain="Clinical NLP",
            tasks=["named entity recognition"],
            target="clinical entities",
            features=["discharge summary text"],
            keywords=["clinical entity", "discharge summary", "ner", "deidentified"],
            dictionary_keywords=["named entity recognition"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "mimic-notes",
        "De-identified Clinical Notes (MIMIC-style)",
        domain="Healthcare",
        tags=["clinical", "ner"],
        description="De-identified discharge summaries annotated with clinical entities.",
        target_description="clinical entity spans",
        feature_description="discharge summary text with entity annotations",
        task_types=["named entity recognition"],
        file_types=["txt", "csv"],
        pub=COLD,
    )
    .add(
        1,
        "news-ner",
        "Newswire Named Entity Corpus",
        domain="Social Media",
        tags=["news", "ner"],
        description="Newswire text annotated with person/location entities (not clinical).",
        target_description="entity spans (person, location, organisation)",
        feature_description="article text",
        task_types=["named entity recognition"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "contracts",
        "Contract Clause Collection",
        domain="Legal",
        tags=["contracts"],
        description="Contract clauses without annotations.",
        task_types=[],
        file_types=["txt"],
        pub=COLD,
    )
    .build(),
    case(
        id="nlp-squad",
        description="Answer extraction from Wikipedia passages (SQuAD-style).",
        category="nlp",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="NLP",
            tasks=["question answering"],
            target="answer span",
            features=["question text", "passage text"],
            keywords=["question answering", "answer span", "wikipedia", "squad"],
            dictionary_keywords=["question answering"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "squad-v1",
        "SQuAD Reading Comprehension (v1)",
        domain="Artificial Intelligence",
        tags=["qa", "squad"],
        description="Wikipedia passages with questions and answer spans.",
        target_description="answer span",
        feature_description="question text and passage text",
        task_types=["question answering"],
        file_types=["json", "txt"],
        pub=HOT,
    )
    .add(
        3,
        "squad-v2",
        "SQuAD v2 with Unanswerable Questions",
        domain="Artificial Intelligence",
        tags=["qa", "squad"],
        description="SQuAD v2 with unanswerable question flags.",
        target_description="answer span / unanswerable flag",
        feature_description="question text and passage text",
        task_types=["question answering"],
        file_types=["json"],
        pub=HOT,
    )
    .add(
        1,
        "summarization",
        "News Headline Summarization Pairs",
        domain="Social Media",
        tags=["news", "summarization"],
        description="Headline-summary pairs (no answer spans).",
        target_description="summary text",
        feature_description="article headline and body",
        task_types=["text summarization"],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="nlp-hate-speech",
        description="Detect hate speech in social media posts.",
        category="nlp",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="NLP",
            tasks=["text classification"],
            target="hate speech",
            features=["post text", "platform"],
            keywords=["hate speech", "toxicity", "posts", "social media"],
            dictionary_keywords=["toxicity detection"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "hate-speech-tweets",
        "Hate Speech Twitter Corpus",
        domain="Social Media",
        tags=["tweets", "hate speech"],
        description="Tweets labelled hate/not-hate with platform metadata.",
        target_description="hate speech label",
        feature_description="post text and platform",
        task_types=["text classification"],
        file_types=["csv", "txt"],
        pub=WARM,
    )
    .add(
        0,
        "imdb-sentiment",
        "IMDB Large Movie Reviews",
        domain="Entertainment",
        tags=["movies", "sentiment"],
        description="Movie review polarity (not hate speech).",
        target_description="sentiment polarity",
        feature_description="review text",
        task_types=["sentiment analysis"],
        file_types=["txt"],
        pub=HOT,
    )
    .build(),
    case(
        id="nlp-sarcasm",
        description="Detect sarcasm in online forum posts.",
        category="nlp",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="NLP",
            tasks=["text classification"],
            target="sarcasm",
            features=["forum post text", "thread"],
            keywords=["sarcasm", "forum", "thread", "reddit"],
            dictionary_keywords=["text classification"],
            data_types=["text"],
        ),
    )
    .add(
        3,
        "reddit-sarcasm",
        "Reddit Sarcasm Corpus",
        domain="Social Media",
        tags=["reddit", "sarcasm"],
        description="Forum posts labelled sarcastic/not-sarcastic with thread structure.",
        target_description="sarcasm label",
        feature_description="forum post text and thread",
        task_types=["text classification"],
        file_types=["json", "txt"],
        pub=WARM,
    )
    .add(
        1,
        "product-reviews-sarcasm",
        "Product Reviews with Sarcasm Notes",
        domain="Retail",
        tags=["reviews"],
        description="Product reviews with occasional sarcasm but no sarcasm labels.",
        target_description="review rating",
        feature_description="review text",
        task_types=["sentiment analysis"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "news-headlines",
        "News Headline Archive",
        domain="Social Media",
        tags=["news"],
        description="News headlines archive without sarcasm labels.",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
]


# ---------------------------------------------------------------------------
# COMPUTER VISION
# ---------------------------------------------------------------------------

COMPUTER_VISION: list[BenchmarkCase] = [
    case(
        id="cv-digit-classification",
        description="Classify handwritten digits from 28x28 grayscale images.",
        category="computer_vision",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="Computer Vision",
            tasks=["image classification"],
            target="digit",
            features=["28x28 grayscale image"],
            keywords=["handwritten digit", "mnist", "28x28", "grayscale"],
            dictionary_keywords=["image classification"],
            data_types=["image data"],
        ),
    )
    .add(
        3,
        "mnist",
        "MNIST Handwritten Digits",
        domain="Artificial Intelligence",
        tags=["digits", "handwriting"],
        description="28x28 grayscale images of handwritten digits labelled 0-9.",
        target_description="digit class (0-9)",
        feature_description="28x28 grayscale image per sample",
        task_types=["image classification"],
        file_types=["csv", "png"],
        pub=HOT,
    )
    .add(
        3,
        "emnist",
        "EMNIST Handwritten Letters and Digits",
        domain="Artificial Intelligence",
        tags=["handwriting", "digits"],
        description="NIST handwritten letters and digits in 28x28 grayscale.",
        target_description="character class",
        feature_description="28x28 grayscale image per sample",
        task_types=["image classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        2,
        "fashion-mnist",
        "Fashion-MNIST",
        domain="Retail",
        tags=["fashion", "images"],
        description="28x28 grayscale images of clothing items.",
        target_description="clothing class",
        feature_description="28x28 grayscale image per sample",
        task_types=["image classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        0,
        "cifar",
        "CIFAR-10 Colour Images",
        domain="Artificial Intelligence",
        tags=["images", "objects"],
        description="32x32 colour images of 10 object categories.",
        target_description="object class",
        task_types=["image classification"],
        file_types=["png"],
        pub=HOT,
    )
    .add(
        -1,
        "iris-tabular",
        "Iris Flower Measurements (Tabular)",
        domain="Environment",
        tags=["flowers"],
        description="Tabular sepal and petal measurements of iris flowers.",
        target_description="iris species",
        task_types=["classification"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="cv-object-detection",
        description="Detect objects with bounding boxes.",
        category="computer_vision",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="Computer Vision",
            tasks=["object detection"],
            target="bounding boxes",
            features=["image", "bounding box coordinates"],
            keywords=["object detection", "bounding box", "images"],
            dictionary_keywords=["object detection"],
            data_types=["image data"],
        ),
    )
    .add(
        3,
        "coco-detection",
        "COCO Detection Annotations",
        domain="Artificial Intelligence",
        tags=["images", "detection"],
        description="Common objects in context with bounding-box annotations.",
        target_description="bounding boxes and object classes",
        feature_description="image with bounding box coordinates per object",
        task_types=["object detection"],
        file_types=["jpg", "json"],
        pub=HOT,
    )
    .add(
        1,
        "cityscapes-seg",
        "Cityscapes Semantic Segmentation",
        domain="Transportation",
        tags=["driving", "segmentation"],
        description="Urban driving images with per-pixel semantic masks (not boxes).",
        target_description="per-pixel semantic class mask",
        feature_description="image with per-pixel segmentation masks",
        task_types=["semantic segmentation"],
        file_types=["png", "json"],
        pub=MID,
    )
    .add(
        0,
        "imagenet",
        "ImageNet Classification (ILSVRC)",
        domain="Artificial Intelligence",
        tags=["images", "classification"],
        description="Image classification labels without boxes.",
        target_description="object class",
        task_types=["image classification"],
        file_types=["jpg"],
        pub=HOT,
    )
    .build(),
    case(
        id="cv-face-detection",
        description="Count faces in images.",
        category="computer_vision",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="Computer Vision",
            tasks=["face detection"],
            target="face count",
            features=["image", "bounding box"],
            keywords=["face", "count", "images", "bounding box"],
            dictionary_keywords=["face detection"],
            data_types=["image data"],
        ),
    )
    .add(
        3,
        "wider-face",
        "WIDER FACE Dataset",
        domain="Artificial Intelligence",
        tags=["faces", "detection"],
        description="Images with dense face annotations for detection and counting.",
        target_description="face bounding boxes and counts",
        feature_description="image with face bounding boxes",
        task_types=["face detection", "object detection"],
        file_types=["jpg", "txt"],
        pub=WARM,
    )
    .add(
        1,
        "celebrity-classification",
        "Celebrity Identity Labels",
        domain="Entertainment",
        tags=["celebrity", "classification"],
        description="Cropped celebrity faces labelled with identity (not count).",
        target_description="identity label",
        task_types=["image classification"],
        file_types=["jpg"],
        pub=WARM,
    )
    .add(
        0,
        "iris-images",
        "Iris Image Database",
        domain="Biometrics",
        tags=["iris"],
        description="Iris images for biometric recognition.",
        target_description="iris identity",
        task_types=["image classification"],
        file_types=["tiff"],
        pub=MID,
    )
    .build(),
    case(
        id="cv-tumor-mri",
        description="Detect brain tumour presence in MRI scans.",
        category="computer_vision",
        requirements=dict(
            domain="Healthcare",
            subdomain="Medical Imaging",
            tasks=["image classification", "detection"],
            target="tumour presence",
            features=["mri scan"],
            keywords=["brain tumour", "mri", "scan", "detection"],
            dictionary_keywords=["tumor detection", "medical image diagnosis"],
            data_types=["image data"],
        ),
    )
    .add(
        3,
        "brain-tumor-mri",
        "Brain Tumour MRI Images",
        domain="Healthcare",
        tags=["mri", "tumour"],
        description="MRI slices labelled by tumour presence and type.",
        target_description="tumour presence and type",
        feature_description="mri scan slice",
        task_types=["image classification"],
        file_types=["png"],
        pub=WARM,
    )
    .add(
        1,
        "chest-xray-pneumonia",
        "Chest X-ray Pneumonia Images",
        domain="Healthcare",
        tags=["x-ray", "pneumonia"],
        description="Paediatric chest X-rays labelled for pneumonia.",
        target_description="pneumonia label",
        feature_description="chest x-ray image",
        task_types=["image classification"],
        file_types=["jpeg"],
        pub=WARM,
    )
    .add(
        1,
        "skin-lesions",
        "Skin Lesion Photos (HAM10000)",
        domain="Healthcare",
        tags=["skin", "lesions"],
        description="Dermoscopic skin lesion images labelled by lesion class.",
        target_description="lesion class",
        feature_description="dermoscopy image",
        task_types=["image classification"],
        file_types=["jpg"],
        pub=WARM,
    )
    .add(
        0,
        "satellite-imagery",
        "Satellite Imagery Tiles",
        domain="Geospatial",
        tags=["satellite"],
        description="Satellite imagery tiles for land cover.",
        target_description="land cover class",
        task_types=["image classification"],
        file_types=["tiff", "png"],
        pub=HOT,
    )
    .build(),
    case(
        id="cv-action-video",
        description="Recognise actions from video clips.",
        category="computer_vision",
        requirements=dict(
            domain="Artificial Intelligence",
            subdomain="Computer Vision",
            tasks=["action recognition"],
            target="action class",
            features=["video clip"],
            keywords=["action recognition", "video", "clip"],
            dictionary_keywords=["action recognition"],
            data_types=["video data"],
        ),
    )
    .add(
        3,
        "ucf101",
        "UCF101 Action Recognition",
        domain="Artificial Intelligence",
        tags=["video", "actions"],
        description="Video clips of 101 action classes.",
        target_description="action class",
        feature_description="video clip per sample",
        task_types=["action recognition"],
        file_types=["avi", "mp4"],
        pub=WARM,
    )
    .add(
        1,
        "still-action",
        "Still Image Action Dataset",
        domain="Artificial Intelligence",
        tags=["images", "actions"],
        description="Single frames labelled with action class (no temporal motion).",
        target_description="action class",
        feature_description="single frame per sample",
        task_types=["image classification"],
        file_types=["jpg"],
        pub=WARM,
    )
    .add(
        0,
        "sports-text",
        "Sports Match Commentary Text",
        domain="Sports Analytics",
        tags=["sports", "text"],
        description="Match commentary text.",
        task_types=[],
        file_types=["txt"],
        pub=COLD,
    )
    .build(),
]


# ---------------------------------------------------------------------------
# TIME SERIES
# ---------------------------------------------------------------------------

TIME_SERIES: list[BenchmarkCase] = [
    case(
        id="ts-solar-output",
        description="Forecast next-hour solar power output from irradiance and weather features.",
        category="time_series",
        requirements=dict(
            domain="Energy",
            subdomain="Renewables",
            tasks=["forecasting"],
            target="solar power output",
            features=["irradiance", "temperature", "timestamp"],
            keywords=["solar", "power output", "irradiance", "forecast"],
            dictionary_keywords=["solar energy prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "solar-plants",
        "Solar Power Plant Generation",
        domain="Energy",
        tags=["solar", "energy"],
        description="Solar plant generation with irradiance and temperature sensors.",
        target_description="solar power output",
        feature_description="irradiance, temperature and timestamp per reading",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "electricity-load",
        "Electricity Load Diagrams",
        domain="Energy",
        tags=["energy", "load"],
        description="Hourly electricity load time series from several zones.",
        target_description="hourly electricity load",
        feature_description="hourly load per zone",
        task_types=["forecasting"],
        file_types=["txt", "csv"],
        pub=WARM,
    )
    .add(
        1,
        "weather-observations",
        "Weather Station Observations",
        domain="Environment",
        tags=["weather"],
        description="Weather observations without energy output labels.",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="ts-airline-passengers",
        description="Forecast monthly airline passenger totals.",
        category="time_series",
        requirements=dict(
            domain="Transportation",
            subdomain="Aviation",
            tasks=["forecasting"],
            target="monthly passengers",
            features=["month", "year", "passenger total"],
            keywords=["airline", "passengers", "monthly", "forecast"],
            dictionary_keywords=["demand forecasting"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "airline-passengers",
        "Airline Passengers Time Series",
        domain="Transportation",
        tags=["airline", "passengers"],
        description="Monthly international airline passenger totals 1949-1960.",
        target_description="monthly passenger total",
        feature_description="month, year and passenger total",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        2,
        "flights-data",
        "Daily Flight Counts",
        domain="Transportation",
        tags=["flights", "airports"],
        description="Daily flight arrivals and departures per airport.",
        target_description="daily flight count",
        feature_description="date, origin, destination and flight count",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="ts-traffic-flow",
        description="Forecast hourly traffic volume at intersections.",
        category="time_series",
        requirements=dict(
            domain="Transportation",
            subdomain="Traffic",
            tasks=["forecasting"],
            target="traffic volume",
            features=["hour", "intersection id", "occupancy rate"],
            keywords=["traffic", "volume", "hourly", "intersection"],
            dictionary_keywords=["traffic prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "traffic-hourly",
        "Hourly Traffic Volume Sensors",
        domain="Transportation",
        tags=["traffic", "sensors"],
        description="Hourly traffic volume and occupancy per intersection sensor.",
        target_description="hourly traffic volume",
        feature_description="hour, intersection id and occupancy rate",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        1,
        "taxi-traffic",
        "Taxi Trip Records",
        domain="Transportation",
        tags=["taxi", "trips"],
        description="Taxi trip records (mobility, not fixed-sensor traffic volume).",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        0,
        "parking-occupancy",
        "Parking Occupancy Sensors",
        domain="Urban Planning",
        tags=["parking"],
        description="Parking bay occupancy sensor readings.",
        task_types=[],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="ts-energy-consumption",
        description="Forecast hourly building energy consumption.",
        category="time_series",
        requirements=dict(
            domain="Energy",
            subdomain="Buildings",
            tasks=["forecasting"],
            target="hourly energy consumption",
            features=["hour", "building id", "outdoor temperature"],
            keywords=["energy consumption", "hourly", "building", "forecast"],
            dictionary_keywords=["energy consumption", "energy demand forecasting"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "building-energy",
        "Building Energy Consumption Hourly",
        domain="Energy",
        tags=["energy", "buildings"],
        description="Hourly energy consumption for several buildings with outdoor temperature.",
        target_description="hourly energy consumption",
        feature_description="hour, building id and outdoor temperature",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        3,
        "electricity-load-hourly",
        "Hourly Electricity Load Time Series",
        domain="Energy",
        tags=["energy", "load"],
        description="Hourly electricity load per zone with timestamps.",
        target_description="hourly electricity consumption",
        feature_description="hour and zone consumption",
        task_types=["forecasting"],
        file_types=["txt", "csv"],
        pub=WARM,
    )
    .add(
        1,
        "natural-gas",
        "Natural Gas Consumption (Monthly)",
        domain="Energy",
        tags=["energy", "gas"],
        description="Monthly natural gas consumption, too coarse for hourly forecasting.",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=COLD,
    )
    .build(),
    case(
        id="ts-market-volatility",
        description="Forecast 30-day market volatility from price history.",
        category="time_series",
        requirements=dict(
            domain="Finance",
            subdomain="Risk",
            tasks=["forecasting", "regression"],
            target="30 day volatility",
            features=["daily returns", "rolling window", "volume"],
            keywords=["volatility", "returns", "market risk", "forecast"],
            dictionary_keywords=["market risk analysis", "stock price prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "options-volatility",
        "Options Implied Volatility Surface",
        domain="Finance",
        tags=["options", "volatility"],
        description="Options implied volatility surfaces with 30-day horizons.",
        target_description="30 day implied volatility",
        feature_description="strike, expiry and daily returns",
        task_types=["regression", "forecasting"],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        2,
        "stock-returns",
        "Daily Stock Returns",
        domain="Finance",
        tags=["stocks", "returns"],
        description="Daily stock returns (volatility must be engineered).",
        target_description="realised volatility",
        feature_description="daily returns and rolling window statistics",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        1,
        "fx-rates",
        "Foreign Exchange Rate History",
        domain="Finance",
        tags=["fx", "rates"],
        description="Foreign exchange rate history across currency pairs.",
        target_description="exchange rate",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
]


# ---------------------------------------------------------------------------
# EDGE CASES - these exist to make the metrics themselves provably safe
# ---------------------------------------------------------------------------

EDGE_CASES: list[BenchmarkCase] = [
    case(
        id="edge-all-unknown-evidence",
        description="Every candidate publishes no target and no features; grades must still be assigned by content.",
        category="edge_case",
        requirements=dict(
            domain="Education",
            subdomain="Assessment",
            tasks=["classification"],
            target="pass fail",
            features=["exam score", "attendance"],
            keywords=["pass fail", "exam score", "attendance"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SPARSE_METADATA,
        notes="UNKNOWN must not collapse into MISMATCH for any candidate here.",
    )
    .add(
        2,
        "silent-passfail",
        "Student Result Summaries",
        domain="Education",
        tags=["education", "results"],
        description="Student result summaries for pass and fail outcomes.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        1,
        "silent-scores",
        "Exam Score Lists",
        domain="Education",
        tags=["education", "scores"],
        description="Exam score lists for enrolled students.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .add(
        0,
        "silent-unrelated",
        "School Bus Routes",
        domain="Transportation",
        tags=["transport", "bus"],
        description="School bus route manifests.",
        task_types=[],
        file_types=["csv"],
        pub=COLD,
    )
    .build(),
    case(
        id="edge-single-candidate",
        description="One candidate only; every metric must be well defined at k=10 for a list of length 1.",
        category="edge_case",
        requirements=dict(
            domain="Retail",
            subdomain="Forecasting",
            tasks=["forecasting"],
            target="weekly units",
            features=["store id", "week"],
            keywords=["weekly units", "store", "forecast"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "only-candidate",
        "Weekly Unit Sales Panel",
        domain="Retail",
        tags=["retail", "sales"],
        description="Weekly units sold per store.",
        target_description="weekly units",
        feature_description="store id and week",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="edge-all-incompatible",
        description="No relevant candidate exists; recall denominators are zero and must not divide by zero.",
        category="edge_case",
        requirements=dict(
            domain="Healthcare",
            subdomain="Genomics",
            tasks=["classification"],
            target="genomic variant pathogenicity",
            features=["variant allele frequency", "clinvar significance"],
            keywords=["genomic variant", "pathogenicity", "clinvar"],
            data_types=["tabular data"],
        ),
        notes="Relevant-in-pool = 0; recall and NDCG must be defined, not NaN.",
    )
    .add(
        -1,
        "retail-sales-only",
        "Retail Seasonality Sales",
        domain="Retail",
        tags=["retail"],
        description="Retail seasonality sales figures.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .add(
        -1,
        "weather-climate",
        "Climate Reanalysis Archive",
        domain="Environment",
        tags=["climate"],
        description="Global climate reanalysis grids.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="edge-domain-only-project",
        description="Project specifies only a domain; every requirement factor is unjudgeable but publisher signals still apply.",
        category="edge_case",
        requirements=dict(
            domain="Cybersecurity",
            subdomain=None,
            tasks=[],
            target=None,
            features=[],
            keywords=["cybersecurity"],
            dictionary_keywords=["network security"],
            data_types=[],
        ),
        notes="Ranks on domain + publisher signals only; must not crash.",
    )
    .add(
        2,
        "domain-only-match",
        "Cybersecurity Incident Archive",
        domain="Cybersecurity",
        tags=["security", "incidents"],
        description="Cybersecurity incident records.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "domain-only-other",
        "Culinary Recipe Archive",
        domain="Food",
        tags=["recipes"],
        description="Recipes with ratings.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="edge-unicode-and-punctuation",
        description="Non-ASCII names and punctuation must rank deterministically and not crash normalisation.",
        category="edge_case",
        requirements=dict(
            domain="Healthcare",
            subdomain="Epidemiology",
            tasks=["forecasting"],
            target="incidence rate",
            features=["date", "district"],
            keywords=["incidence", "district", "epidemic"],
            dictionary_keywords=["disease prediction"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "unicode-correct",
        "Taux d'incidence —district sanitaire",
        domain="Healthcare",
        tags=["épidémiologie", "incidence"],
        description="Incidence rate by district and date (é sanitation districts).",
        target_description="incidence rate",
        feature_description="date and district",
        task_types=["forecasting"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "unicode-distractor",
        "Recettes de cuisine :: données",
        domain="Food",
        tags=["recettes"],
        description="Cuisine recipe dataset with punctuation-heavy names.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
    case(
        id="edge-ai-rationale-trap",
        description="Only the AI rationale mentions the required target; a ranker matching rationale would invert this case.",
        category="edge_case",
        requirements=dict(
            domain="Environment",
            subdomain="Air Quality",
            tasks=["regression"],
            target="pm2.5 concentration",
            features=["pm2.5", "wind speed", "timestamp"],
            keywords=["pm2.5", "air quality", "wind speed"],
            dictionary_keywords=["air quality monitoring"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.SEMANTIC,
        notes="rationale text names pm2.5 concentration; declared fields do not.",
    )
    .add(
        1,
        "ai-pm25-rationale",
        "Urban Sensing Programme Data",
        description="Measurements from fixed-site urban sensing monitors.",
        task_types=[],
        file_types=["csv"],
        ai=True,
        possible_source="Urban sensing programme",
        pub=UNPUBLISHED,
    )
    .ai_rationale(
        "ai-pm25-rationale",
        "This dataset contains pm2.5 concentration, wind speed and timestamp columns, matching the project exactly.",
    )
    .add(
        2,
        "declared-pm25",
        "PM2.5 Air Quality Monitor Records",
        domain="Environment",
        tags=["air quality", "pm2.5"],
        description="Air quality monitor records with pm2.5 concentration and wind speed.",
        target_description="pm2.5 concentration",
        feature_description="columns include pm2.5, wind speed and timestamp",
        task_types=["regression"],
        file_types=["csv"],
        pub=MID,
    )
    .build(),
    case(
        id="edge-retrieval-miss",
        description="The pool contains no grade-3 dataset: a retrieval failure, not a ranking failure.",
        category="edge_case",
        requirements=dict(
            domain="Agriculture",
            subdomain="Precision Agriculture",
            tasks=["regression"],
            target="crop yield",
            features=["soil moisture", "rainfall", "crop type"],
            keywords=["crop yield", "soil moisture", "rainfall"],
            dictionary_keywords=["crop yield prediction"],
            data_types=["tabular data"],
        ),
        difficulty=Difficulty.RETRIEVAL,
        retrieval_complete=False,
        notes="Metric ceiling is bounded by the pool; classification must say RETRIEVAL.",
    )
    .add(
        1,
        "regional-harvest",
        "Regional Harvest Statistics",
        domain="Agriculture",
        tags=["harvest", "statistics"],
        description="Regional harvest totals by season without soil or rainfall features.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .add(
        0,
        "weather-history",
        "Weather History Archive",
        domain="Environment",
        tags=["weather"],
        description="Historical weather observations.",
        task_types=[],
        file_types=["csv"],
        pub=WARM,
    )
    .build(),
    case(
        id="edge-dense-and-identical",
        description="Two near-identical candidates must rank deterministically with no hidden ordering by object identity.",
        category="edge_case",
        requirements=dict(
            domain="Finance",
            subdomain="Risk",
            tasks=["classification"],
            target="default",
            features=["income", "duration"],
            keywords=["default", "income", "duration"],
            dictionary_keywords=["credit scoring"],
            data_types=["tabular data"],
        ),
    )
    .add(
        3,
        "credit-a",
        "Credit Risk Applicants (Part A)",
        domain="Finance",
        tags=["credit", "risk"],
        description="Applicant records with income and duration labelled for default.",
        target_description="default label",
        feature_description="income and duration",
        task_types=["classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        3,
        "credit-b",
        "Credit Risk Applicants (Part B)",
        domain="Finance",
        tags=["credit", "risk"],
        description="Applicant records with income and duration labelled for default.",
        target_description="default label",
        feature_description="income and duration",
        task_types=["classification"],
        file_types=["csv"],
        pub=MID,
    )
    .add(
        0,
        "credit-generic",
        "Consumer Finance Overview",
        domain="Finance",
        tags=["finance"],
        description="Consumer finance overview statistics.",
        task_types=[],
        file_types=["csv"],
        pub=HOT,
    )
    .build(),
]


def all_cases() -> list[BenchmarkCase]:
    """Every benchmark case, in a stable order."""
    return [
        *EDUCATION,
        *HEALTHCARE,
        *CYBERSECURITY,
        *FINANCE,
        *RETAIL,
        *NLP,
        *COMPUTER_VISION,
        *TIME_SERIES,
        *EDGE_CASES,
    ]


def load_benchmark() -> list[BenchmarkCase]:
    """Load and validate the benchmark corpus.

    Validation is part of loading rather than a test, because a malformed case
    (duplicate id, duplicate candidate key, ungraded candidate, grade outside the
    scale) must never reach a metric: silently dropping one would make a metric
    look like it improved.
    """
    cases = all_cases()
    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            raise ValueError(f"duplicate benchmark case id: {case.id!r}")
        seen_ids.add(case.id)
        keys = [spec.key for spec in case.candidates]
        if len(set(keys)) != len(keys):
            raise ValueError(f"case {case.id!r} has duplicate candidate keys")
        for spec in case.candidates:
            grade = case.grade_of(spec.key)
            if grade not in (-1, 0, 1, 2, 3):
                raise ValueError(
                    f"case {case.id!r} candidate {spec.key!r} has grade {grade} "
                    "outside {-1, 0, 1, 2, 3}"
                )
        if not case.candidates:
            raise ValueError(f"case {case.id!r} has no candidates")
    if len(cases) < 50:
        raise ValueError(f"benchmark has only {len(cases)} cases; 50 is the floor")
    return cases


__all__ = [
    "BENCHMARK_VERSION",
    "EDGE_CASES",
    "all_cases",
    "load_benchmark",
]