/**
 * Friendly names for the backend contracts. Everything is derived from the generated `api.ts`
 * (see `npm run gen:api`), so a backend change that breaks the UI fails the type-check instead of the user.
 *
 * OpenAPI marks fields that have defaults as optional, but the backend always serialises them, so response
 * types are wrapped in `DeepRequired` (a `| null` stays `| null`: unknown is still unknown).
 */
import type { components } from "./api";

type S = components["schemas"];

type DeepRequired<T> = T extends (infer U)[]
  ? DeepRequired<U>[]
  : T extends object
    ? { [K in keyof T]-?: DeepRequired<T[K]> }
    : T;

export type AssistResponse = DeepRequired<S["AssistResponse"]>;
export type IntentResolution = DeepRequired<S["IntentResolution"]>;
export type NextAction = DeepRequired<S["NextAction"]>;
export type DatasetRecommendationResponse = DeepRequired<S["DatasetRecommendationResponse"]>;
export type RankedDataset = DeepRequired<S["RankedDataset"]>;
export type FinalRecommendation = DeepRequired<S["FinalRecommendation"]>;
export type RecommendedDataset = DeepRequired<S["RecommendedDataset"]>;
export type DatasetIntelligence = DeepRequired<S["DatasetIntelligence"]>;
export type DatasetRisk = DeepRequired<S["DatasetRisk"]>;
export type FeatureCompatibility = DeepRequired<S["FeatureCompatibility"]>;
export type SourceOutcome = DeepRequired<S["SourceOutcome"]>;

export type DatasetAnalysisResponse = DeepRequired<S["DatasetAnalysisResponse"]>;
export type DatasetAnalysis = DeepRequired<S["DatasetAnalysis"]>;
export type ColumnProfile = DeepRequired<S["ColumnProfile"]>;
export type QualityFinding = DeepRequired<S["QualityFinding"]>;
export type TargetCandidate = DeepRequired<S["TargetCandidate"]>;
export type TaskCandidate = DeepRequired<S["TaskCandidate"]>;

export type ProjectIntelligence = DeepRequired<S["ProjectIntelligence"]>;
export type FeatureAssessment = DeepRequired<S["FeatureAssessment"]>;
export type FeatureIntelligence = DeepRequired<S["FeatureIntelligence"]>;
export type ProjectIdea = DeepRequired<S["ProjectIdea"]>;
export type ModelRecommendation = DeepRequired<S["ModelRecommendation"]>;
export type ProjectCompatibility = DeepRequired<S["ProjectCompatibility"]>;

/** The research request is an *input*: its optional fields really are optional. */
export type ResearchRequest = S["ResearchRequest"];
export type ResearchResponse = DeepRequired<S["ResearchResponse"]>;
export type Paper = DeepRequired<S["Paper"]>;
export type MethodSupport = DeepRequired<S["MethodSupport"]>;
export type ReportedMetric = DeepRequired<S["ReportedMetric"]>;
export type AlgorithmRecommendation = DeepRequired<S["AlgorithmRecommendation"]>;

export type ReadyResponse = DeepRequired<S["ReadyResponse"]>;
export type UserIntent = S["UserIntent"];
