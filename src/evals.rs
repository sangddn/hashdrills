// Copyright 2026 Sang Doan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Reproducible, provider-neutral model evaluation.
//!
//! Both runners exercise the same [`ModelBackend`]
//! and [`LlmBackend`] paths used by practice
//! sessions. Reports omit authored case text and model feedback unless the
//! caller explicitly requests generated samples.

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs::File;
use std::io::ErrorKind;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use serde::Deserialize;
use serde::Serialize;
use tokio::task::JoinSet;

use crate::model::DEFAULT_TIMEOUT;
use crate::model::Evaluation;
use crate::model::GeneratedInstance;
use crate::model::GenerationQualityJudgment;
use crate::model::LlmBackend;
use crate::model::ModelBackend;
use crate::model::PROTOCOL_VERSION;
use crate::model::Verdict;
use crate::spec::DrillSpec;
use crate::spec::Template;

/// Schema version for machine-readable reports emitted by this module.
pub const EVAL_REPORT_VERSION: u32 = 1;

/// Maximum accepted size of a custom evaluation suite.
pub const MAX_EVAL_SUITE_BYTES: u64 = 1_048_576;

/// Maximum candidate or answer calls in one invocation.
pub const MAX_EVAL_CALLS: usize = 1_000;

/// Maximum combined rendered Q/target/rubric bytes retained per candidate.
pub const MAX_RETAINED_GENERATION_BYTES: usize = 65_536;

const EVALUATION_CASES_JSON: &str = include_str!("../evals/evaluation_cases.json");
const GENERATION_CASES_JSON: &str = include_str!("../evals/generation_cases.json");

/// One human-labelled answer-evaluation case.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCase {
    pub id: String,
    pub tags: Vec<String>,
    pub question: String,
    pub criteria: String,
    pub response: String,
    pub allowed_verdicts: Vec<Verdict>,
}

/// One authored generation-quality case.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationCase {
    pub id: String,
    pub domain: String,
    pub goal: Option<String>,
    pub question_template: String,
    pub answer_template: String,
    pub judge_requirements: String,
}

impl GenerationCase {
    /// Parse this case into the same stable specification used in production.
    pub fn to_spec(&self) -> Result<DrillSpec, EvalError> {
        let question = Template::parse(&self.question_template).map_err(|error| {
            EvalError::invalid_suite(format!(
                "generation case `{}` has an invalid question template: {error}",
                self.id
            ))
        })?;
        let answer = Template::parse(&self.answer_template).map_err(|error| {
            EvalError::invalid_suite(format!(
                "generation case `{}` has an invalid answer template: {error}",
                self.id
            ))
        })?;
        Ok(DrillSpec::new(
            "generation-eval",
            None,
            PathBuf::from("generation-eval.md"),
            (1, 1),
            self.goal.clone(),
            question,
            answer,
        ))
    }
}

/// Shared selection and parallelism controls for an evaluation run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalRunConfig {
    /// Number of calls per selected case, from 1 through 10.
    pub repeats: usize,
    /// Maximum simultaneous calls, from 1 through 16.
    pub concurrency: usize,
    /// Exact case IDs. An empty vector selects the entire suite.
    pub case_ids: Vec<String>,
    /// Seed mixed with case ID and repeat number for generation variation.
    pub seed: u64,
    /// Independent runner deadline for every backend call.
    pub call_timeout: Duration,
}

impl Default for EvalRunConfig {
    fn default() -> Self {
        Self {
            repeats: 1,
            concurrency: 1,
            case_ids: Vec::new(),
            seed: 0,
            call_timeout: DEFAULT_TIMEOUT,
        }
    }
}

/// Additional controls for generation evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationEvalConfig {
    pub run: EvalRunConfig,
    /// Maximum simultaneous judge calls, from 1 through 16.
    pub judge_concurrency: usize,
    /// Return generation and diversity data without calling a judge.
    pub skip_judge: bool,
    /// Include rendered Q/A/rubric samples in the report.
    pub include_samples: bool,
}

impl Default for GenerationEvalConfig {
    fn default() -> Self {
        Self {
            run: EvalRunConfig::default(),
            judge_concurrency: 1,
            skip_judge: false,
            include_samples: false,
        }
    }
}

/// A validation, loading, or run-level evaluation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalError {
    message: String,
}

impl EvalError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn invalid_suite(message: impl Into<String>) -> Self {
        Self::new(format!("invalid evaluation suite: {}", message.into()))
    }

    fn invalid_config(message: impl Into<String>) -> Self {
        Self::new(format!(
            "invalid evaluation configuration: {}",
            message.into()
        ))
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for EvalError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for EvalError {}

/// Counts for all five protocol verdicts.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct VerdictDistribution {
    pub pass: usize,
    pub partial: usize,
    pub fail: usize,
    pub uncertain: usize,
    pub invalid: usize,
}

/// Correct calls and their percentage of usable calls.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AccuracySummary {
    pub correct: usize,
    pub total: usize,
    pub percent: f64,
}

/// Nearest-rank latency summary in wall-clock milliseconds.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct LatencySummary {
    pub samples: usize,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub max_ms: u64,
}

/// Aggregate feedback size without retaining feedback text.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct FeedbackMetrics {
    pub samples: usize,
    pub total_chars: usize,
    pub mean_chars: f64,
    pub p50_chars: u64,
    pub p95_chars: u64,
    pub max_chars: u64,
    pub total_words: usize,
    pub mean_words: f64,
    pub p50_words: u64,
    pub p95_words: u64,
    pub max_words: u64,
}

/// Private-safe outcome for one answer-evaluation call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EvaluationOutcome {
    pub verdict: Verdict,
    pub binary_correct: bool,
    pub exact_correct: bool,
}

/// Private-safe record for one answer-evaluation call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EvaluationCallReport {
    pub case_id: String,
    /// One-based repeat number.
    pub repeat: usize,
    pub outcome: Option<EvaluationOutcome>,
    pub elapsed_ms: u64,
    pub error: Option<String>,
}

/// Answer-evaluation results grouped by a case's first tag.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationDomainSummary {
    pub domain: String,
    pub calls: usize,
    pub usable_calls: usize,
    pub errors: usize,
    pub binary_accuracy: AccuracySummary,
    pub exact_accuracy: AccuracySummary,
    pub false_yea: usize,
    pub false_nay: usize,
}

/// Versioned machine-readable answer-evaluation report.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationReport {
    pub report_version: u32,
    pub calls: usize,
    pub usable_calls: usize,
    pub errors: usize,
    pub models: Vec<String>,
    pub verdicts: VerdictDistribution,
    pub binary_accuracy: AccuracySummary,
    pub exact_accuracy: AccuracySummary,
    pub false_yea: usize,
    pub false_nay: usize,
    pub latency: LatencySummary,
    pub feedback: FeedbackMetrics,
    pub domains: Vec<EvaluationDomainSummary>,
    pub results: Vec<EvaluationCallReport>,
}

/// Counts of failed generation-quality dimensions.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct QualityDimensionFailures {
    pub question_invalid: usize,
    pub target_incorrect: usize,
    pub alignment: usize,
    pub constraints: usize,
    pub answer_leakage: usize,
    pub atomicity: usize,
}

/// Lexical diversity across generated questions.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LexicalDiversity {
    pub samples: usize,
    pub unique_questions: usize,
    pub duplicate_questions: usize,
    pub unique_percent: f64,
    /// Mean pairwise Jaccard distance between token sets within each case.
    pub mean_pairwise_distance: Option<f64>,
}

/// Deliberately opt-in generated material attached to a call report.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GenerationSample {
    pub question: String,
    pub target: String,
    pub rubric: String,
}

/// Outcome of candidate generation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationOutcome {
    Success,
    Error,
}

/// Outcome of the optional blind quality check.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeOutcome {
    Pass,
    Fail,
    Error,
    Skipped,
    NotRun,
}

/// Private-safe record for one generation-evaluation call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GenerationCallReport {
    pub case_id: String,
    /// One-based repeat number.
    pub repeat: usize,
    pub generation: GenerationOutcome,
    pub judge: JudgeOutcome,
    pub generation_ms: u64,
    pub judge_ms: Option<u64>,
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<GenerationSample>,
}

/// Versioned machine-readable generation-evaluation report.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GenerationReport {
    pub report_version: u32,
    pub calls: usize,
    pub usable_generations: usize,
    pub generation_errors: usize,
    pub generation_success_percent: f64,
    pub candidate_models: Vec<String>,
    pub generation_latency: LatencySummary,
    pub judge_requested: bool,
    /// `correct` is the number judged and `total` is usable generations.
    pub judge_coverage: AccuracySummary,
    pub judge_errors: usize,
    pub quality_passes: usize,
    pub quality_percent_of_judged: f64,
    pub end_to_end_quality_percent: f64,
    pub judge_models: Vec<String>,
    pub dimension_failures: QualityDimensionFailures,
    pub judge_latency: LatencySummary,
    pub lexical_diversity: LexicalDiversity,
    pub results: Vec<GenerationCallReport>,
}

/// Load and validate the evaluation cases shipped with Hashdrills.
pub fn bundled_evaluation_cases() -> Result<Vec<EvaluationCase>, EvalError> {
    parse_evaluation_cases(EVALUATION_CASES_JSON, "bundled evaluation cases")
}

/// Load and validate the generation cases shipped with Hashdrills.
pub fn bundled_generation_cases() -> Result<Vec<GenerationCase>, EvalError> {
    parse_generation_cases(GENERATION_CASES_JSON, "bundled generation cases")
}

/// Load a strict custom answer-evaluation JSON array.
pub fn load_evaluation_cases(path: &Path) -> Result<Vec<EvaluationCase>, EvalError> {
    let source = read_suite(path)?;
    parse_evaluation_cases(&source, &path.display().to_string())
}

/// Load a strict custom generation-evaluation JSON array.
pub fn load_generation_cases(path: &Path) -> Result<Vec<GenerationCase>, EvalError> {
    let source = read_suite(path)?;
    parse_generation_cases(&source, &path.display().to_string())
}

fn parse_evaluation_cases(
    source: &str,
    description: &str,
) -> Result<Vec<EvaluationCase>, EvalError> {
    let cases: Vec<EvaluationCase> = serde_json::from_str(source).map_err(|error| {
        EvalError::invalid_suite(format!(
            "could not parse {description} as JSON: {}",
            safe_suite_json_error(error)
        ))
    })?;
    validate_evaluation_cases(&cases)?;
    Ok(cases)
}

fn parse_generation_cases(
    source: &str,
    description: &str,
) -> Result<Vec<GenerationCase>, EvalError> {
    let cases: Vec<GenerationCase> = serde_json::from_str(source).map_err(|error| {
        EvalError::invalid_suite(format!(
            "could not parse {description} as JSON: {}",
            safe_suite_json_error(error)
        ))
    })?;
    validate_generation_cases(&cases)?;
    Ok(cases)
}

fn safe_suite_json_error(error: serde_json::Error) -> String {
    let category = match error.classify() {
        serde_json::error::Category::Io => "JSON could not be read",
        serde_json::error::Category::Syntax => "JSON syntax was malformed",
        serde_json::error::Category::Data => "JSON did not match the suite schema",
        serde_json::error::Category::Eof => "JSON ended unexpectedly",
    };
    format!(
        "{category} at line {}, column {}",
        error.line(),
        error.column()
    )
}

fn read_suite(path: &Path) -> Result<String, EvalError> {
    let link_metadata = std::fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
    if link_metadata.file_type().is_symlink() || !link_metadata.is_file() {
        return Err(EvalError::invalid_suite(format!(
            "{} must be a regular file, not a symlink",
            path.display()
        )));
    }
    if link_metadata.len() > MAX_EVAL_SUITE_BYTES {
        return Err(suite_too_large(path));
    }

    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let metadata = file.metadata().map_err(|error| io_error(path, error))?;
    if !metadata.is_file() {
        return Err(EvalError::invalid_suite(format!(
            "{} must be a regular file",
            path.display()
        )));
    }
    let mut bytes = Vec::new();
    file.take(MAX_EVAL_SUITE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    if bytes.len() as u64 > MAX_EVAL_SUITE_BYTES {
        return Err(suite_too_large(path));
    }
    String::from_utf8(bytes)
        .map_err(|_| EvalError::invalid_suite(format!("{} is not valid UTF-8", path.display())))
}

fn io_error(path: &Path, error: std::io::Error) -> EvalError {
    let detail = match error.kind() {
        ErrorKind::NotFound => "does not exist",
        ErrorKind::PermissionDenied => "is not readable",
        _ => "could not be read",
    };
    EvalError::invalid_suite(format!("{} {detail}", path.display()))
}

fn suite_too_large(path: &Path) -> EvalError {
    EvalError::invalid_suite(format!(
        "{} exceeds the {} byte limit",
        path.display(),
        MAX_EVAL_SUITE_BYTES
    ))
}

fn validate_evaluation_cases(cases: &[EvaluationCase]) -> Result<(), EvalError> {
    validate_nonempty_suite(cases.len())?;
    let mut ids = HashSet::new();
    for case in cases {
        validate_label(&case.id, "case ID")?;
        if !ids.insert(case.id.as_str()) {
            return Err(EvalError::invalid_suite(format!(
                "duplicate case ID `{}`",
                case.id
            )));
        }
        validate_text(&case.question, &case.id, "question")?;
        validate_text(&case.criteria, &case.id, "criteria")?;
        validate_labels(&case.tags, &case.id, "tag")?;
        if case.allowed_verdicts.is_empty() {
            return Err(EvalError::invalid_suite(format!(
                "case `{}` has no allowed verdicts",
                case.id
            )));
        }
        let mut labels = HashSet::new();
        for verdict in &case.allowed_verdicts {
            if !labels.insert(*verdict) {
                return Err(EvalError::invalid_suite(format!(
                    "case `{}` repeats allowed verdict `{verdict}`",
                    case.id
                )));
            }
        }
        if case.allowed_verdicts.contains(&Verdict::Pass) && case.allowed_verdicts.len() != 1 {
            return Err(EvalError::invalid_suite(format!(
                "case `{}` mixes the YEA label `pass` with NAY labels",
                case.id
            )));
        }
    }
    Ok(())
}

fn validate_generation_cases(cases: &[GenerationCase]) -> Result<(), EvalError> {
    validate_nonempty_suite(cases.len())?;
    let mut ids = HashSet::new();
    for case in cases {
        validate_label(&case.id, "case ID")?;
        if !ids.insert(case.id.as_str()) {
            return Err(EvalError::invalid_suite(format!(
                "duplicate case ID `{}`",
                case.id
            )));
        }
        validate_label(&case.domain, "domain")?;
        if let Some(goal) = &case.goal {
            validate_text(goal, &case.id, "goal")?;
        }
        validate_text(&case.question_template, &case.id, "question template")?;
        validate_text(&case.answer_template, &case.id, "answer template")?;
        validate_text(&case.judge_requirements, &case.id, "judge requirements")?;
        let spec = case.to_spec()?;
        if spec.question.directive_count() == 0 {
            return Err(EvalError::invalid_suite(format!(
                "case `{}` needs at least one Q directive",
                case.id
            )));
        }
        if spec.answer.directive_count() == 0 {
            return Err(EvalError::invalid_suite(format!(
                "case `{}` needs at least one A directive",
                case.id
            )));
        }
    }
    Ok(())
}

fn validate_nonempty_suite(length: usize) -> Result<(), EvalError> {
    if length == 0 {
        Err(EvalError::invalid_suite("the case array is empty"))
    } else {
        Ok(())
    }
}

fn validate_text(value: &str, case_id: &str, label: &str) -> Result<(), EvalError> {
    if value.trim().is_empty() {
        Err(EvalError::invalid_suite(format!(
            "case `{case_id}` has blank {label}"
        )))
    } else {
        Ok(())
    }
}

fn validate_label(value: &str, label: &str) -> Result<(), EvalError> {
    if value.trim().is_empty() {
        return Err(EvalError::invalid_suite(format!("blank {label}")));
    }
    if value.trim() != value {
        return Err(EvalError::invalid_suite(format!(
            "{label} `{value}` has surrounding whitespace"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(EvalError::invalid_suite(format!(
            "{label} contains a control character"
        )));
    }
    Ok(())
}

fn validate_labels(values: &[String], case_id: &str, label: &str) -> Result<(), EvalError> {
    if values.is_empty() {
        return Err(EvalError::invalid_suite(format!(
            "case `{case_id}` has no {label}s"
        )));
    }
    let mut seen = HashSet::new();
    for value in values {
        validate_label(value, label)?;
        if !seen.insert(value.as_str()) {
            return Err(EvalError::invalid_suite(format!(
                "case `{case_id}` repeats {label} `{value}`"
            )));
        }
    }
    Ok(())
}

fn validate_run_config(config: &EvalRunConfig) -> Result<(), EvalError> {
    if !(1..=10).contains(&config.repeats) {
        return Err(EvalError::invalid_config(
            "repeats must be from 1 through 10",
        ));
    }
    if !(1..=16).contains(&config.concurrency) {
        return Err(EvalError::invalid_config(
            "concurrency must be from 1 through 16",
        ));
    }
    if config.call_timeout.is_zero() {
        return Err(EvalError::invalid_config(
            "call timeout must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_call_count(
    case_count: usize,
    repeats: usize,
    kind: &'static str,
    judge_requested: bool,
) -> Result<usize, EvalError> {
    let calls = case_count
        .checked_mul(repeats)
        .ok_or_else(|| EvalError::invalid_config("case count times repeats is too large"))?;
    if calls > MAX_EVAL_CALLS {
        let judge = if judge_requested {
            format!(" and up to {calls} judge calls")
        } else {
            String::new()
        };
        return Err(EvalError::invalid_config(format!(
            "the run requests {calls} {kind} calls{judge}; the limit is {MAX_EVAL_CALLS} {kind} calls"
        )));
    }
    Ok(calls)
}

fn select_cases<T: Clone>(
    cases: &[T],
    requested: &[String],
    id: impl Fn(&T) -> &str,
) -> Result<Vec<T>, EvalError> {
    if requested.is_empty() {
        return Ok(cases.to_vec());
    }
    let mut by_id = BTreeMap::new();
    for case in cases {
        by_id.insert(id(case), case);
    }
    let mut seen = HashSet::new();
    let mut selected = Vec::with_capacity(requested.len());
    for requested_id in requested {
        validate_label(requested_id, "selected case ID")
            .map_err(|error| EvalError::invalid_config(error.message))?;
        if !seen.insert(requested_id.as_str()) {
            return Err(EvalError::invalid_config(format!(
                "case ID `{requested_id}` was selected more than once"
            )));
        }
        let Some(case) = by_id.get(requested_id.as_str()) else {
            return Err(EvalError::invalid_config(format!(
                "unknown case ID `{requested_id}`"
            )));
        };
        selected.push((*case).clone());
    }
    Ok(selected)
}

/// Derive a stable generation variation key from a seed, case, and repeat.
pub fn deterministic_variation_key(seed: u64, case_id: &str, repeat: usize) -> u64 {
    let mut input = Vec::with_capacity(case_id.len() + 16);
    input.extend_from_slice(b"hashdrills:eval-variation:v1\0");
    input.extend_from_slice(&seed.to_le_bytes());
    input.extend_from_slice(&(repeat as u64).to_le_bytes());
    input.extend_from_slice(case_id.as_bytes());
    let digest = blake3::hash(&input);
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

/// Run answer-evaluation cases through the exact production evaluation path.
pub async fn run_evaluation_suite(
    cases: &[EvaluationCase],
    backend: Arc<dyn ModelBackend>,
    config: EvalRunConfig,
) -> Result<EvaluationReport, EvalError> {
    validate_evaluation_cases(cases)?;
    validate_run_config(&config)?;
    let selected = select_cases(cases, &config.case_ids, |case| &case.id)?;
    let call_count = validate_call_count(selected.len(), config.repeats, "answer", false)?;
    let mut jobs = VecDeque::new();
    for (case_order, case) in selected.into_iter().enumerate() {
        for repeat in 1..=config.repeats {
            jobs.push_back((case_order, repeat, case.clone()));
        }
    }

    let mut tasks = JoinSet::new();
    let mut records = Vec::with_capacity(call_count);
    fill_evaluation_tasks(
        &mut tasks,
        &mut jobs,
        config.concurrency,
        config.call_timeout,
        &backend,
    );
    while let Some(joined) = tasks.join_next().await {
        let record = match joined {
            Ok(record) => record,
            Err(_) => {
                tasks.shutdown().await;
                return Err(EvalError::new("an evaluation worker failed"));
            }
        };
        records.push(record);
        fill_evaluation_tasks(
            &mut tasks,
            &mut jobs,
            config.concurrency,
            config.call_timeout,
            &backend,
        );
    }
    records.sort_by_key(|record| (record.case_order, record.repeat));
    if !records.iter().any(|record| record.evaluation.is_some()) {
        return Err(EvalError::new(
            "evaluation produced zero usable model results",
        ));
    }
    Ok(summarize_evaluations(&records))
}

#[derive(Clone, Debug)]
struct EvaluationRecord {
    case_order: usize,
    case_id: String,
    domain: String,
    repeat: usize,
    allowed_verdicts: Vec<Verdict>,
    elapsed_ms: u64,
    evaluation: Option<Evaluation>,
    error: Option<String>,
}

fn fill_evaluation_tasks(
    tasks: &mut JoinSet<EvaluationRecord>,
    jobs: &mut VecDeque<(usize, usize, EvaluationCase)>,
    concurrency: usize,
    call_timeout: Duration,
    backend: &Arc<dyn ModelBackend>,
) {
    while tasks.len() < concurrency {
        let Some((case_order, repeat, case)) = jobs.pop_front() else {
            break;
        };
        let backend = Arc::clone(backend);
        tasks.spawn(async move {
            let instance = GeneratedInstance {
                question: case.question,
                target: case.criteria.clone(),
                rubric: case.criteria,
                model: "eval-fixture".into(),
                protocol_version: PROTOCOL_VERSION,
            };
            let started = Instant::now();
            let result =
                tokio::time::timeout(call_timeout, backend.evaluate(&instance, &case.response))
                    .await;
            let elapsed_ms = elapsed_ms(started);
            match result {
                Ok(Ok(evaluation)) => EvaluationRecord {
                    case_order,
                    case_id: case.id,
                    domain: case.tags[0].clone(),
                    repeat,
                    allowed_verdicts: case.allowed_verdicts,
                    elapsed_ms,
                    evaluation: Some(evaluation),
                    error: None,
                },
                Ok(Err(error)) => EvaluationRecord {
                    case_order,
                    case_id: case.id,
                    domain: case.tags[0].clone(),
                    repeat,
                    allowed_verdicts: case.allowed_verdicts,
                    elapsed_ms,
                    evaluation: None,
                    error: Some(error.to_string()),
                },
                Err(_) => EvaluationRecord {
                    case_order,
                    case_id: case.id,
                    domain: case.tags[0].clone(),
                    repeat,
                    allowed_verdicts: case.allowed_verdicts,
                    elapsed_ms,
                    evaluation: None,
                    error: Some(timeout_error(call_timeout)),
                },
            }
        });
    }
}

fn summarize_evaluations(records: &[EvaluationRecord]) -> EvaluationReport {
    let usable: Vec<&EvaluationRecord> = records
        .iter()
        .filter(|record| record.evaluation.is_some())
        .collect();
    let mut models: Vec<String> = usable
        .iter()
        .filter_map(|record| record.evaluation.as_ref())
        .map(|evaluation| evaluation.model.clone())
        .collect();
    models.sort();
    models.dedup();

    let binary_correct_count = usable
        .iter()
        .filter(|record| binary_correct(record))
        .count();
    let exact_correct_count = usable.iter().filter(|record| exact_correct(record)).count();
    let false_yea = usable.iter().filter(|record| false_yea(record)).count();
    let false_nay = usable.iter().filter(|record| false_nay(record)).count();
    let verdicts = verdict_distribution(&usable);
    let latency_values: Vec<u64> = records.iter().map(|record| record.elapsed_ms).collect();
    let feedback_values: Vec<&str> = usable
        .iter()
        .filter_map(|record| record.evaluation.as_ref())
        .map(|evaluation| evaluation.feedback.as_str())
        .collect();

    EvaluationReport {
        report_version: EVAL_REPORT_VERSION,
        calls: records.len(),
        usable_calls: usable.len(),
        errors: records.len().saturating_sub(usable.len()),
        models,
        verdicts,
        binary_accuracy: accuracy(binary_correct_count, usable.len()),
        exact_accuracy: accuracy(exact_correct_count, usable.len()),
        false_yea,
        false_nay,
        latency: latency_summary(&latency_values),
        feedback: feedback_metrics(&feedback_values),
        domains: evaluation_domain_summaries(records),
        results: records
            .iter()
            .map(|record| EvaluationCallReport {
                case_id: record.case_id.clone(),
                repeat: record.repeat,
                outcome: record
                    .evaluation
                    .as_ref()
                    .map(|evaluation| EvaluationOutcome {
                        verdict: evaluation.verdict,
                        binary_correct: binary_correct(record),
                        exact_correct: exact_correct(record),
                    }),
                elapsed_ms: record.elapsed_ms,
                error: record.error.clone(),
            })
            .collect(),
    }
}

fn verdict_distribution(records: &[&EvaluationRecord]) -> VerdictDistribution {
    let mut distribution = VerdictDistribution::default();
    for verdict in records
        .iter()
        .filter_map(|record| record.evaluation.as_ref())
        .map(|evaluation| evaluation.verdict)
    {
        match verdict {
            Verdict::Pass => distribution.pass += 1,
            Verdict::Partial => distribution.partial += 1,
            Verdict::Fail => distribution.fail += 1,
            Verdict::Uncertain => distribution.uncertain += 1,
            Verdict::Invalid => distribution.invalid += 1,
        }
    }
    distribution
}

fn binary_correct(record: &EvaluationRecord) -> bool {
    let expected_yea = record.allowed_verdicts.contains(&Verdict::Pass);
    let actual_yea = record
        .evaluation
        .as_ref()
        .is_some_and(|evaluation| evaluation.verdict == Verdict::Pass);
    expected_yea == actual_yea
}

fn exact_correct(record: &EvaluationRecord) -> bool {
    record
        .evaluation
        .as_ref()
        .is_some_and(|evaluation| record.allowed_verdicts.contains(&evaluation.verdict))
}

fn false_yea(record: &EvaluationRecord) -> bool {
    record
        .evaluation
        .as_ref()
        .is_some_and(|evaluation| evaluation.verdict == Verdict::Pass)
        && !record.allowed_verdicts.contains(&Verdict::Pass)
}

fn false_nay(record: &EvaluationRecord) -> bool {
    record
        .evaluation
        .as_ref()
        .is_some_and(|evaluation| evaluation.verdict != Verdict::Pass)
        && record.allowed_verdicts.contains(&Verdict::Pass)
}

fn evaluation_domain_summaries(records: &[EvaluationRecord]) -> Vec<EvaluationDomainSummary> {
    let mut grouped: BTreeMap<&str, Vec<&EvaluationRecord>> = BTreeMap::new();
    for record in records {
        grouped.entry(&record.domain).or_default().push(record);
    }
    grouped
        .into_iter()
        .map(|(domain, records)| {
            let usable: Vec<&EvaluationRecord> = records
                .iter()
                .copied()
                .filter(|record| record.evaluation.is_some())
                .collect();
            let binary = usable
                .iter()
                .filter(|record| binary_correct(record))
                .count();
            let exact = usable.iter().filter(|record| exact_correct(record)).count();
            EvaluationDomainSummary {
                domain: domain.to_string(),
                calls: records.len(),
                usable_calls: usable.len(),
                errors: records.len().saturating_sub(usable.len()),
                binary_accuracy: accuracy(binary, usable.len()),
                exact_accuracy: accuracy(exact, usable.len()),
                false_yea: usable.iter().filter(|record| false_yea(record)).count(),
                false_nay: usable.iter().filter(|record| false_nay(record)).count(),
            }
        })
        .collect()
}

/// Run candidate generation first, then optionally judge all usable results.
pub async fn run_generation_suite(
    cases: &[GenerationCase],
    candidate: Arc<LlmBackend>,
    judge: Option<Arc<LlmBackend>>,
    config: GenerationEvalConfig,
) -> Result<GenerationReport, EvalError> {
    validate_generation_cases(cases)?;
    validate_run_config(&config.run)?;
    if !(1..=16).contains(&config.judge_concurrency) {
        return Err(EvalError::invalid_config(
            "judge concurrency must be from 1 through 16",
        ));
    }
    if !config.skip_judge && judge.is_none() {
        return Err(EvalError::invalid_config(
            "a separate judge backend is required unless judging is skipped",
        ));
    }
    let selected = select_cases(cases, &config.run.case_ids, |case| &case.id)?;
    let call_count = validate_call_count(
        selected.len(),
        config.run.repeats,
        "candidate",
        !config.skip_judge,
    )?;
    let mut jobs = VecDeque::new();
    for (case_order, case) in selected.into_iter().enumerate() {
        for repeat in 1..=config.run.repeats {
            jobs.push_back((case_order, repeat, case.clone()));
        }
    }

    let mut tasks = JoinSet::new();
    let mut records = Vec::with_capacity(call_count);
    fill_generation_tasks(
        &mut tasks,
        &mut jobs,
        config.run.concurrency,
        config.run.seed,
        config.run.call_timeout,
        &candidate,
    );
    while let Some(joined) = tasks.join_next().await {
        let record = match joined {
            Ok(record) => record,
            Err(_) => {
                tasks.shutdown().await;
                return Err(EvalError::new("a generation worker failed"));
            }
        };
        records.push(record);
        fill_generation_tasks(
            &mut tasks,
            &mut jobs,
            config.run.concurrency,
            config.run.seed,
            config.run.call_timeout,
            &candidate,
        );
    }
    records.sort_by_key(|record| (record.case_order, record.repeat));
    if !records.iter().any(|record| record.instance.is_some()) {
        return Err(EvalError::new(
            "generation evaluation produced zero usable candidate results",
        ));
    }

    // This second phase starts only after every candidate call has completed,
    // keeping judge traffic out of generation latency.
    if !config.skip_judge {
        run_generation_judges(
            &mut records,
            judge.expect("judge was validated"),
            config.judge_concurrency,
            config.run.call_timeout,
        )
        .await?;
        if !records.iter().any(|record| record.judgment.is_some()) {
            return Err(EvalError::new(
                "generation evaluation produced zero usable judge results",
            ));
        }
    }

    Ok(summarize_generations(&records, &config))
}

#[derive(Clone, Debug)]
struct GenerationRecord {
    case_order: usize,
    case: GenerationCase,
    repeat: usize,
    generation_ms: u64,
    instance: Option<GeneratedInstance>,
    generation_error: Option<String>,
    judgment: Option<GenerationQualityJudgment>,
    judge_ms: Option<u64>,
    judge_error: Option<String>,
}

fn fill_generation_tasks(
    tasks: &mut JoinSet<GenerationRecord>,
    jobs: &mut VecDeque<(usize, usize, GenerationCase)>,
    concurrency: usize,
    seed: u64,
    call_timeout: Duration,
    candidate: &Arc<LlmBackend>,
) {
    while tasks.len() < concurrency {
        let Some((case_order, repeat, case)) = jobs.pop_front() else {
            break;
        };
        let candidate = Arc::clone(candidate);
        tasks.spawn(async move {
            let variation_key = deterministic_variation_key(seed, &case.id, repeat);
            let spec = match case.to_spec() {
                Ok(spec) => spec,
                Err(error) => {
                    return GenerationRecord {
                        case_order,
                        case,
                        repeat,
                        generation_ms: 0,
                        instance: None,
                        generation_error: Some(error.to_string()),
                        judgment: None,
                        judge_ms: None,
                        judge_error: None,
                    };
                }
            };
            let started = Instant::now();
            let result = tokio::time::timeout(
                call_timeout,
                candidate.generate_with_variation(&spec, variation_key),
            )
            .await;
            let generation_ms = elapsed_ms(started);
            match result {
                Ok(Ok(instance)) if generated_instance_fits(&instance) => GenerationRecord {
                    case_order,
                    case,
                    repeat,
                    generation_ms,
                    instance: Some(instance),
                    generation_error: None,
                    judgment: None,
                    judge_ms: None,
                    judge_error: None,
                },
                Ok(Ok(_)) => GenerationRecord {
                    case_order,
                    case,
                    repeat,
                    generation_ms,
                    instance: None,
                    generation_error: Some(format!(
                        "generated instance exceeded the {MAX_RETAINED_GENERATION_BYTES} byte eval retention limit"
                    )),
                    judgment: None,
                    judge_ms: None,
                    judge_error: None,
                },
                Ok(Err(error)) => GenerationRecord {
                    case_order,
                    case,
                    repeat,
                    generation_ms,
                    instance: None,
                    generation_error: Some(error.to_string()),
                    judgment: None,
                    judge_ms: None,
                    judge_error: None,
                },
                Err(_) => GenerationRecord {
                    case_order,
                    case,
                    repeat,
                    generation_ms,
                    instance: None,
                    generation_error: Some(timeout_error(call_timeout)),
                    judgment: None,
                    judge_ms: None,
                    judge_error: None,
                },
            }
        });
    }
}

fn generated_instance_fits(instance: &GeneratedInstance) -> bool {
    instance
        .question
        .len()
        .saturating_add(instance.target.len())
        .saturating_add(instance.rubric.len())
        <= MAX_RETAINED_GENERATION_BYTES
}

async fn run_generation_judges(
    records: &mut [GenerationRecord],
    judge: Arc<LlmBackend>,
    concurrency: usize,
    call_timeout: Duration,
) -> Result<(), EvalError> {
    let mut jobs: VecDeque<usize> = records
        .iter()
        .enumerate()
        .filter_map(|(index, record)| record.instance.as_ref().map(|_| index))
        .collect();
    let mut tasks = JoinSet::new();
    fill_judge_tasks(
        &mut tasks,
        &mut jobs,
        concurrency,
        call_timeout,
        records,
        &judge,
    );
    while let Some(joined) = tasks.join_next().await {
        let (index, elapsed, result) = match joined {
            Ok(result) => result,
            Err(_) => {
                tasks.shutdown().await;
                return Err(EvalError::new("a generation judge worker failed"));
            }
        };
        records[index].judge_ms = Some(elapsed);
        match result {
            Ok(judgment) => records[index].judgment = Some(judgment),
            Err(error) => records[index].judge_error = Some(error),
        }
        fill_judge_tasks(
            &mut tasks,
            &mut jobs,
            concurrency,
            call_timeout,
            records,
            &judge,
        );
    }
    Ok(())
}

fn fill_judge_tasks(
    tasks: &mut JoinSet<(usize, u64, Result<GenerationQualityJudgment, String>)>,
    jobs: &mut VecDeque<usize>,
    concurrency: usize,
    call_timeout: Duration,
    records: &[GenerationRecord],
    judge: &Arc<LlmBackend>,
) {
    while tasks.len() < concurrency {
        let Some(index) = jobs.pop_front() else {
            break;
        };
        let judge = Arc::clone(judge);
        let case = records[index].case.clone();
        let instance = records[index]
            .instance
            .clone()
            .expect("judge jobs contain only usable generations");
        tasks.spawn(async move {
            let spec = match case.to_spec() {
                Ok(spec) => spec,
                Err(error) => return (index, 0, Err(error.to_string())),
            };
            let started = Instant::now();
            let result = match tokio::time::timeout(
                call_timeout,
                judge.judge_generation_quality(&spec, &case.judge_requirements, &instance),
            )
            .await
            {
                Ok(result) => result.map_err(|error| error.to_string()),
                Err(_) => Err(timeout_error(call_timeout)),
            };
            (index, elapsed_ms(started), result)
        });
    }
}

fn summarize_generations(
    records: &[GenerationRecord],
    config: &GenerationEvalConfig,
) -> GenerationReport {
    let usable: Vec<&GenerationRecord> = records
        .iter()
        .filter(|record| record.instance.is_some())
        .collect();
    let judged: Vec<&GenerationRecord> = usable
        .iter()
        .copied()
        .filter(|record| record.judgment.is_some())
        .collect();
    let quality_passes = judged
        .iter()
        .filter(|record| {
            record
                .judgment
                .as_ref()
                .is_some_and(GenerationQualityJudgment::is_quality_pass)
        })
        .count();
    let generation_latencies: Vec<u64> =
        records.iter().map(|record| record.generation_ms).collect();
    let judge_latencies: Vec<u64> = usable.iter().filter_map(|record| record.judge_ms).collect();
    let mut candidate_models: Vec<String> = usable
        .iter()
        .filter_map(|record| record.instance.as_ref())
        .map(|instance| instance.model.clone())
        .collect();
    candidate_models.sort();
    candidate_models.dedup();
    let mut judge_models: Vec<String> = judged
        .iter()
        .filter_map(|record| record.judgment.as_ref())
        .map(|judgment| judgment.model.clone())
        .collect();
    judge_models.sort();
    judge_models.dedup();

    GenerationReport {
        report_version: EVAL_REPORT_VERSION,
        calls: records.len(),
        usable_generations: usable.len(),
        generation_errors: records.len().saturating_sub(usable.len()),
        generation_success_percent: percent(usable.len(), records.len()),
        candidate_models,
        generation_latency: latency_summary(&generation_latencies),
        judge_requested: !config.skip_judge,
        judge_coverage: accuracy(judged.len(), usable.len()),
        judge_errors: if config.skip_judge {
            0
        } else {
            usable.len().saturating_sub(judged.len())
        },
        quality_passes,
        quality_percent_of_judged: percent(quality_passes, judged.len()),
        end_to_end_quality_percent: percent(quality_passes, records.len()),
        judge_models,
        dimension_failures: quality_dimension_failures(&judged),
        judge_latency: latency_summary(&judge_latencies),
        lexical_diversity: lexical_diversity(&usable),
        results: records
            .iter()
            .map(|record| generation_call_report(record, config))
            .collect(),
    }
}

fn quality_dimension_failures(records: &[&GenerationRecord]) -> QualityDimensionFailures {
    let mut failures = QualityDimensionFailures::default();
    for judgment in records.iter().filter_map(|record| record.judgment.as_ref()) {
        if !judgment.question_valid {
            failures.question_invalid += 1;
        }
        if !judgment.target_correct {
            failures.target_incorrect += 1;
        }
        if !judgment.aligned {
            failures.alignment += 1;
        }
        if !judgment.constraints_met {
            failures.constraints += 1;
        }
        if judgment.answer_leaked {
            failures.answer_leakage += 1;
        }
        if !judgment.atomic {
            failures.atomicity += 1;
        }
    }
    failures
}

fn generation_call_report(
    record: &GenerationRecord,
    config: &GenerationEvalConfig,
) -> GenerationCallReport {
    let generation = if record.instance.is_some() {
        GenerationOutcome::Success
    } else {
        GenerationOutcome::Error
    };
    let judge = if config.skip_judge {
        JudgeOutcome::Skipped
    } else if record.instance.is_none() {
        JudgeOutcome::NotRun
    } else if let Some(judgment) = &record.judgment {
        if judgment.is_quality_pass() {
            JudgeOutcome::Pass
        } else {
            JudgeOutcome::Fail
        }
    } else {
        JudgeOutcome::Error
    };
    let error = record
        .generation_error
        .clone()
        .or_else(|| record.judge_error.clone());
    let sample = config.include_samples.then(|| {
        record.instance.as_ref().map(|instance| GenerationSample {
            question: instance.question.clone(),
            target: instance.target.clone(),
            rubric: instance.rubric.clone(),
        })
    });
    GenerationCallReport {
        case_id: record.case.id.clone(),
        repeat: record.repeat,
        generation,
        judge,
        generation_ms: record.generation_ms,
        judge_ms: record.judge_ms,
        error,
        sample: sample.flatten(),
    }
}

fn lexical_diversity(records: &[&GenerationRecord]) -> LexicalDiversity {
    let mut by_case: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for record in records {
        if let Some(instance) = &record.instance {
            by_case
                .entry(&record.case.id)
                .or_default()
                .push(normalize_question(&instance.question));
        }
    }
    let samples: usize = by_case.values().map(Vec::len).sum();
    let unique_questions: usize = by_case
        .values()
        .map(|questions| questions.iter().collect::<HashSet<_>>().len())
        .sum();
    let duplicate_questions = samples.saturating_sub(unique_questions);
    let mut distances = Vec::new();
    for questions in by_case.values() {
        for left in 0..questions.len() {
            for right in (left + 1)..questions.len() {
                distances.push(jaccard_distance(&questions[left], &questions[right]));
            }
        }
    }
    LexicalDiversity {
        samples,
        unique_questions,
        duplicate_questions,
        unique_percent: percent(unique_questions, samples),
        mean_pairwise_distance: if distances.is_empty() {
            None
        } else {
            Some(distances.iter().sum::<f64>() / distances.len() as f64)
        },
    }
}

fn normalize_question(question: &str) -> String {
    question
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn jaccard_distance(left: &str, right: &str) -> f64 {
    let left: HashSet<&str> = left.split_whitespace().collect();
    let right: HashSet<&str> = right.split_whitespace().collect();
    let union = left.union(&right).count();
    if union == 0 {
        0.0
    } else {
        1.0 - left.intersection(&right).count() as f64 / union as f64
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn timeout_error(timeout: Duration) -> String {
    format!(
        "model call exceeded the {:.3} second eval deadline",
        timeout.as_secs_f64()
    )
}

fn percent(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 * 100.0 / denominator as f64
    }
}

fn accuracy(correct: usize, total: usize) -> AccuracySummary {
    AccuracySummary {
        correct,
        total,
        percent: percent(correct, total),
    }
}

fn latency_summary(values: &[u64]) -> LatencySummary {
    LatencySummary {
        samples: values.len(),
        p50_ms: nearest_rank_percentile(values, 50),
        p95_ms: nearest_rank_percentile(values, 95),
        max_ms: values.iter().copied().max().unwrap_or(0),
    }
}

fn feedback_metrics(feedback: &[&str]) -> FeedbackMetrics {
    let chars: Vec<u64> = feedback
        .iter()
        .map(|value| value.chars().count() as u64)
        .collect();
    let words: Vec<u64> = feedback
        .iter()
        .map(|value| value.split_whitespace().count() as u64)
        .collect();
    let total_chars = chars.iter().sum::<u64>() as usize;
    let total_words = words.iter().sum::<u64>() as usize;
    FeedbackMetrics {
        samples: feedback.len(),
        total_chars,
        mean_chars: mean(total_chars, feedback.len()),
        p50_chars: nearest_rank_percentile(&chars, 50),
        p95_chars: nearest_rank_percentile(&chars, 95),
        max_chars: chars.iter().copied().max().unwrap_or(0),
        total_words,
        mean_words: mean(total_words, feedback.len()),
        p50_words: nearest_rank_percentile(&words, 50),
        p95_words: nearest_rank_percentile(&words, 95),
        max_words: words.iter().copied().max().unwrap_or(0),
    }
}

fn mean(total: usize, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

/// Return a nearest-rank percentile from an unsorted sample.
pub fn nearest_rank_percentile(values: &[u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let percentile = percentile.clamp(1, 100);
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (percentile * sorted.len()).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use serde_json::Value;

    use super::*;
    use crate::model::ModelError;
    use crate::model::ModelFuture;

    struct FakeBackend {
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl FakeBackend {
        fn new() -> Self {
            Self {
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
            }
        }
    }

    impl ModelBackend for FakeBackend {
        fn generate<'a>(&'a self, _spec: &'a DrillSpec) -> ModelFuture<'a, GeneratedInstance> {
            Box::pin(async { Err(ModelError::InvalidConfiguration("unused fake generation")) })
        }

        fn evaluate<'a>(
            &'a self,
            _instance: &'a GeneratedInstance,
            response: &'a str,
        ) -> ModelFuture<'a, Evaluation> {
            let response = response.to_string();
            Box::pin(async move {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_active.fetch_max(active, Ordering::SeqCst);
                if response.contains("slow") {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                } else {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                self.active.fetch_sub(1, Ordering::SeqCst);
                if response.contains("pending") {
                    std::future::pending::<()>().await;
                }
                if response.contains("backend-error") {
                    return Err(ModelError::BackendFailed(Some(9)));
                }
                let verdict = if response.contains("partial") {
                    Verdict::Partial
                } else if response.contains("fail") {
                    Verdict::Fail
                } else {
                    Verdict::Pass
                };
                Ok(Evaluation {
                    verdict,
                    feedback: "SECRET_MODEL_FEEDBACK".into(),
                    model: "fake-model".into(),
                    protocol_version: PROTOCOL_VERSION,
                })
            })
        }
    }

    fn evaluation_case(
        id: &str,
        domain: &str,
        response: &str,
        allowed_verdicts: Vec<Verdict>,
    ) -> EvaluationCase {
        EvaluationCase {
            id: id.into(),
            tags: vec![domain.into()],
            question: "SECRET_QUESTION".into(),
            criteria: "SECRET_CRITERIA".into(),
            response: format!("SECRET_RESPONSE {response}"),
            allowed_verdicts,
        }
    }

    fn many_evaluation_cases(count: usize) -> Vec<EvaluationCase> {
        (0..count)
            .map(|index| {
                evaluation_case(
                    &format!("case-{index}"),
                    "math",
                    "pass",
                    vec![Verdict::Pass],
                )
            })
            .collect()
    }

    fn generation_case() -> GenerationCase {
        GenerationCase {
            id: "generation-case".into(),
            domain: "math".into(),
            goal: Some("SECRET_GOAL".into()),
            question_template: "What is {{choose one digit}}?".into(),
            answer_template: "{{give the same digit}}".into(),
            judge_requirements: "Q and A contain the same digit.".into(),
        }
    }

    fn many_generation_cases(count: usize) -> Vec<GenerationCase> {
        (0..count)
            .map(|index| GenerationCase {
                id: format!("generation-{index}"),
                ..generation_case()
            })
            .collect()
    }

    #[test]
    fn bundled_suites_are_strict_and_well_formed() {
        let evaluation = bundled_evaluation_cases().unwrap();
        let generation = bundled_generation_cases().unwrap();
        assert!(evaluation.len() >= 60);
        assert!(generation.len() >= 12);
        assert!(generation.iter().all(|case| {
            let spec = case.to_spec().unwrap();
            spec.question.directive_count() > 0 && spec.answer.directive_count() > 0
        }));
    }

    #[test]
    fn custom_loader_rejects_unknown_fields_and_oversize_files() {
        let mut unknown = tempfile::NamedTempFile::new().unwrap();
        write!(
            unknown,
            r#"[{{"id":"x","tags":["math"],"question":"q","criteria":"c","response":"r","allowed_verdicts":["pass"],"extra":true}}]"#
        )
        .unwrap();
        let error = load_evaluation_cases(unknown.path()).unwrap_err();
        assert!(error.to_string().contains("did not match the suite schema"));
        assert!(!error.to_string().contains("extra"));

        let oversized = tempfile::NamedTempFile::new().unwrap();
        oversized
            .as_file()
            .set_len(MAX_EVAL_SUITE_BYTES + 1)
            .unwrap();
        let error = load_generation_cases(oversized.path()).unwrap_err();
        assert!(error.to_string().contains("byte limit"));
    }

    #[cfg(unix)]
    #[test]
    fn custom_loader_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.json");
        std::fs::write(&target, "[]").unwrap();
        let link = directory.path().join("link.json");
        symlink(target, &link).unwrap();
        let error = load_evaluation_cases(&link).unwrap_err();
        assert!(error.to_string().contains("not a symlink"));
    }

    #[test]
    fn suite_validation_rejects_duplicate_and_mixed_labels() {
        let mut duplicate_tags = evaluation_case("x", "math", "pass", vec![Verdict::Pass]);
        duplicate_tags.tags.push("math".into());
        let error = validate_evaluation_cases(&[duplicate_tags]).unwrap_err();
        assert!(error.to_string().contains("repeats tag"));

        let mixed = evaluation_case("x", "math", "pass", vec![Verdict::Pass, Verdict::Partial]);
        let error = validate_evaluation_cases(&[mixed]).unwrap_err();
        assert!(error.to_string().contains("mixes the YEA label"));

        let duplicated = vec![
            evaluation_case("x", "math", "pass", vec![Verdict::Pass]),
            evaluation_case("x", "math", "fail", vec![Verdict::Fail]),
        ];
        assert!(
            validate_evaluation_cases(&duplicated)
                .unwrap_err()
                .to_string()
                .contains("duplicate case ID")
        );
    }

    #[test]
    fn suite_validation_allows_a_blank_learner_response() {
        let case = evaluation_case("blank", "practical", "pass", vec![Verdict::Invalid]);
        let mut case = case;
        case.response = "  \n".into();
        validate_evaluation_cases(&[case]).unwrap();
    }

    #[test]
    fn generation_validation_requires_directives_in_both_fields() {
        let mut case = generation_case();
        case.question_template = "Static question".into();
        let error = validate_generation_cases(&[case]).unwrap_err();
        assert!(error.to_string().contains("Q directive"));
    }

    #[test]
    fn case_selection_is_exact_ordered_and_rejects_duplicates() {
        let cases = vec![
            evaluation_case("a", "math", "pass", vec![Verdict::Pass]),
            evaluation_case("b", "math", "pass", vec![Verdict::Pass]),
        ];
        let selected = select_cases(&cases, &["b".into(), "a".into()], |case| &case.id).unwrap();
        assert_eq!(selected[0].id, "b");
        assert_eq!(selected[1].id, "a");
        assert!(
            select_cases(&cases, &["a".into(), "a".into()], |case| &case.id)
                .unwrap_err()
                .to_string()
                .contains("more than once")
        );
        assert!(
            select_cases(&cases, &["missing".into()], |case| &case.id)
                .unwrap_err()
                .to_string()
                .contains("unknown case ID")
        );
    }

    #[test]
    fn run_configuration_bounds_are_strict() {
        for repeats in [0, 11] {
            let config = EvalRunConfig {
                repeats,
                ..EvalRunConfig::default()
            };
            assert!(validate_run_config(&config).is_err());
        }
        for concurrency in [0, 17] {
            let config = EvalRunConfig {
                concurrency,
                ..EvalRunConfig::default()
            };
            assert!(validate_run_config(&config).is_err());
        }
        let config = EvalRunConfig {
            call_timeout: Duration::ZERO,
            ..EvalRunConfig::default()
        };
        assert!(validate_run_config(&config).is_err());
    }

    #[test]
    fn call_limit_accepts_boundary_and_describes_candidate_and_judge_traffic() {
        assert_eq!(
            validate_call_count(MAX_EVAL_CALLS / 10, 10, "answer", false).unwrap(),
            MAX_EVAL_CALLS
        );
        let error = validate_call_count(MAX_EVAL_CALLS / 10 + 1, 10, "candidate", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("1010 candidate calls"));
        assert!(error.contains("1010 judge calls"));
        assert!(error.contains("limit is 1000 candidate calls"));
    }

    #[tokio::test]
    async fn answer_runner_rejects_over_limit_but_counts_selected_subset_only() {
        let cases = many_evaluation_cases(101);
        let backend = Arc::new(FakeBackend::new());
        let error = run_evaluation_suite(
            &cases,
            backend.clone(),
            EvalRunConfig {
                repeats: 10,
                concurrency: 16,
                case_ids: Vec::new(),
                seed: 0,
                call_timeout: DEFAULT_TIMEOUT,
            },
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("1010 answer calls"));
        assert_eq!(backend.max_active.load(Ordering::SeqCst), 0);

        let report = run_evaluation_suite(
            &cases,
            backend,
            EvalRunConfig {
                repeats: 10,
                concurrency: 16,
                case_ids: vec!["case-100".into(), "case-0".into()],
                seed: 0,
                call_timeout: DEFAULT_TIMEOUT,
            },
        )
        .await
        .unwrap();
        assert_eq!(report.calls, 20);
        assert_eq!(report.results[0].case_id, "case-100");
        assert_eq!(report.results[10].case_id, "case-0");
    }

    #[tokio::test]
    async fn generation_runner_rejects_over_limit_before_backend_use() {
        let cases = many_generation_cases(101);
        let error = run_generation_suite(
            &cases,
            Arc::new(LlmBackend::new("unused-model")),
            Some(Arc::new(LlmBackend::new("unused-judge"))),
            GenerationEvalConfig {
                run: EvalRunConfig {
                    repeats: 10,
                    concurrency: 16,
                    case_ids: Vec::new(),
                    seed: 0,
                    call_timeout: DEFAULT_TIMEOUT,
                },
                judge_concurrency: 16,
                skip_judge: false,
                include_samples: false,
            },
        )
        .await
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("1010 candidate calls"));
        assert!(message.contains("1010 judge calls"));
    }

    #[test]
    fn oversized_generated_instances_are_not_retained() {
        let within = GeneratedInstance {
            question: "q".repeat(MAX_RETAINED_GENERATION_BYTES - 2),
            target: "a".into(),
            rubric: "r".into(),
            model: "fake".into(),
            protocol_version: PROTOCOL_VERSION,
        };
        assert!(generated_instance_fits(&within));
        let oversized = GeneratedInstance {
            question: "q".repeat(MAX_RETAINED_GENERATION_BYTES - 1),
            ..within
        };
        assert!(!generated_instance_fits(&oversized));
    }

    #[tokio::test]
    async fn answer_runner_scores_orders_limits_and_serializes_privately() {
        let backend = Arc::new(FakeBackend::new());
        let cases = vec![
            evaluation_case("a", "math", "slow pass", vec![Verdict::Pass]),
            evaluation_case("b", "math", "pass", vec![Verdict::Fail]),
            evaluation_case("c", "physics", "partial", vec![Verdict::Pass]),
            evaluation_case(
                "d",
                "physics",
                "partial",
                vec![Verdict::Partial, Verdict::Fail],
            ),
            evaluation_case("e", "physics", "backend-error", vec![Verdict::Fail]),
        ];
        let report = run_evaluation_suite(
            &cases,
            backend.clone(),
            EvalRunConfig {
                repeats: 1,
                concurrency: 3,
                case_ids: Vec::new(),
                seed: 42,
                call_timeout: DEFAULT_TIMEOUT,
            },
        )
        .await
        .unwrap();
        assert_eq!(report.report_version, EVAL_REPORT_VERSION);
        assert_eq!(report.calls, 5);
        assert_eq!(report.usable_calls, 4);
        assert_eq!(report.errors, 1);
        assert_eq!(report.binary_accuracy, accuracy(2, 4));
        assert_eq!(report.exact_accuracy, accuracy(2, 4));
        assert_eq!(report.false_yea, 1);
        assert_eq!(report.false_nay, 1);
        assert_eq!(report.verdicts.pass, 2);
        assert_eq!(report.verdicts.partial, 2);
        assert_eq!(report.models, vec!["fake-model"]);
        assert_eq!(
            report
                .results
                .iter()
                .map(|result| result.case_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c", "d", "e"]
        );
        assert!(backend.max_active.load(Ordering::SeqCst) > 1);
        assert!(backend.max_active.load(Ordering::SeqCst) <= 3);

        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("SECRET_QUESTION"));
        assert!(!json.contains("SECRET_CRITERIA"));
        assert!(!json.contains("SECRET_RESPONSE"));
        assert!(!json.contains("SECRET_MODEL_FEEDBACK"));
        serde_json::from_str::<EvaluationReport>(&json).unwrap();
    }

    #[tokio::test]
    async fn answer_runner_errors_when_every_call_is_unusable() {
        let cases = vec![evaluation_case(
            "a",
            "math",
            "backend-error",
            vec![Verdict::Fail],
        )];
        let error = run_evaluation_suite(
            &cases,
            Arc::new(FakeBackend::new()),
            EvalRunConfig::default(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("zero usable"));
    }

    #[tokio::test]
    async fn answer_runner_deadlines_a_pending_backend_call() {
        let cases = vec![
            evaluation_case("pending", "math", "pending", vec![Verdict::Fail]),
            evaluation_case("usable", "math", "pass", vec![Verdict::Pass]),
        ];
        let report = run_evaluation_suite(
            &cases,
            Arc::new(FakeBackend::new()),
            EvalRunConfig {
                concurrency: 2,
                call_timeout: Duration::from_millis(20),
                ..EvalRunConfig::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(report.calls, 2);
        assert_eq!(report.usable_calls, 1);
        assert_eq!(report.errors, 1);
        assert!(
            report.results[0]
                .error
                .as_deref()
                .unwrap()
                .contains("eval deadline")
        );
    }

    #[test]
    fn variation_keys_are_seeded_stable_and_repeat_specific() {
        let first = deterministic_variation_key(7, "case", 1);
        assert_eq!(first, deterministic_variation_key(7, "case", 1));
        assert_ne!(first, deterministic_variation_key(8, "case", 1));
        assert_ne!(first, deterministic_variation_key(7, "case", 2));
        assert_ne!(first, deterministic_variation_key(7, "other", 1));
    }

    #[test]
    fn nearest_rank_percentiles_are_centralized_and_exact() {
        assert_eq!(nearest_rank_percentile(&[], 50), 0);
        assert_eq!(nearest_rank_percentile(&[4, 1, 3, 2], 50), 2);
        assert_eq!(nearest_rank_percentile(&[4, 1, 3, 2], 95), 4);
        assert_eq!(nearest_rank_percentile(&[1, 2], 50), 1);
    }

    #[test]
    fn lexical_diversity_normalizes_cosmetic_changes() {
        let case = generation_case();
        let make_record = |repeat, question: &str| GenerationRecord {
            case_order: 0,
            case: case.clone(),
            repeat,
            generation_ms: 1,
            instance: Some(GeneratedInstance {
                question: question.into(),
                target: "answer".into(),
                rubric: "answer".into(),
                model: "fake".into(),
                protocol_version: PROTOCOL_VERSION,
            }),
            generation_error: None,
            judgment: None,
            judge_ms: None,
            judge_error: None,
        };
        let records = [
            make_record(1, "What is 3 + 4?"),
            make_record(2, "WHAT IS 3 + 4"),
            make_record(3, "Compute 8 minus 2"),
        ];
        let refs = records.iter().collect::<Vec<_>>();
        let diversity = lexical_diversity(&refs);
        assert_eq!(diversity.samples, 3);
        assert_eq!(diversity.unique_questions, 2);
        assert_eq!(diversity.duplicate_questions, 1);
        assert!(diversity.mean_pairwise_distance.unwrap() > 0.0);
    }

    #[test]
    fn generation_report_omits_material_without_explicit_samples() {
        let case = generation_case();
        let records = vec![GenerationRecord {
            case_order: 0,
            case,
            repeat: 1,
            generation_ms: 12,
            instance: Some(GeneratedInstance {
                question: "SECRET_GENERATED_QUESTION".into(),
                target: "SECRET_GENERATED_TARGET".into(),
                rubric: "SECRET_GENERATED_RUBRIC".into(),
                model: "candidate".into(),
                protocol_version: PROTOCOL_VERSION,
            }),
            generation_error: None,
            judgment: Some(GenerationQualityJudgment {
                question_valid: true,
                target_correct: false,
                aligned: false,
                constraints_met: true,
                answer_leaked: true,
                atomic: false,
                feedback: "SECRET_JUDGE_FEEDBACK".into(),
                model: "judge".into(),
                protocol_version: PROTOCOL_VERSION,
            }),
            judge_ms: Some(7),
            judge_error: None,
        }];
        let report = summarize_generations(&records, &GenerationEvalConfig::default());
        assert_eq!(report.quality_passes, 0);
        assert_eq!(report.dimension_failures.target_incorrect, 1);
        assert_eq!(report.dimension_failures.alignment, 1);
        assert_eq!(report.dimension_failures.answer_leakage, 1);
        assert_eq!(report.dimension_failures.atomicity, 1);
        assert_eq!(report.results[0].judge, JudgeOutcome::Fail);
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("SECRET_GENERATED"));
        assert!(!json.contains("SECRET_JUDGE_FEEDBACK"));
        serde_json::from_str::<GenerationReport>(&json).unwrap();
    }

    #[cfg(unix)]
    fn fake_llm_script(directory: &Path, name: &str, output: &str, exit: i32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = directory.join(name);
        let mut file = File::create(&path).unwrap();
        writeln!(file, "#!/bin/sh").unwrap();
        writeln!(file, "cat >/dev/null").unwrap();
        writeln!(file, "printf '%s' '{}'", output.replace('\'', "'\\''")).unwrap();
        writeln!(file, "exit {exit}").unwrap();
        let mut permissions = file.metadata().unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn generation_runner_finishes_candidates_before_blind_judging() {
        let directory = tempfile::tempdir().unwrap();
        let candidate_path = fake_llm_script(
            directory.path(),
            "candidate",
            r#"{"question_replacements":["7"],"answer_replacements":["7"]}"#,
            0,
        );
        let judge_path = fake_llm_script(
            directory.path(),
            "judge",
            r#"{"question_valid":true,"target_correct":true,"aligned":true,"constraints_met":true,"answer_leaked":false,"atomic":true,"feedback":"OK"}"#,
            0,
        );
        let candidate =
            Arc::new(LlmBackend::new("candidate-model").with_executable(candidate_path));
        let judge = Arc::new(
            LlmBackend::new("judge-model")
                .with_evaluation_model("fixed-judge")
                .with_executable(judge_path),
        );
        let report = run_generation_suite(
            &[generation_case()],
            candidate,
            Some(judge),
            GenerationEvalConfig {
                run: EvalRunConfig {
                    repeats: 2,
                    concurrency: 2,
                    case_ids: Vec::new(),
                    seed: 123,
                    call_timeout: DEFAULT_TIMEOUT,
                },
                judge_concurrency: 2,
                skip_judge: false,
                include_samples: true,
            },
        )
        .await
        .unwrap();
        assert_eq!(report.calls, 2);
        assert_eq!(report.usable_generations, 2);
        assert_eq!(report.judge_coverage, accuracy(2, 2));
        assert_eq!(report.quality_passes, 2);
        assert_eq!(report.end_to_end_quality_percent, 100.0);
        assert_eq!(report.candidate_models, ["candidate-model"]);
        assert_eq!(report.judge_models, ["fixed-judge"]);
        assert!(
            report
                .results
                .iter()
                .all(|result| result.sample.as_ref().unwrap().question == "What is 7?")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn generation_runner_requires_usable_candidate_and_judge_results() {
        let directory = tempfile::tempdir().unwrap();
        let failed = fake_llm_script(directory.path(), "failed", "", 1);
        let candidate_error = run_generation_suite(
            &[generation_case()],
            Arc::new(LlmBackend::new("candidate").with_executable(&failed)),
            None,
            GenerationEvalConfig {
                skip_judge: true,
                ..GenerationEvalConfig::default()
            },
        )
        .await
        .unwrap_err();
        assert!(
            candidate_error
                .to_string()
                .contains("zero usable candidate")
        );

        let candidate_path = fake_llm_script(
            directory.path(),
            "candidate-ok",
            r#"{"question_replacements":["4"],"answer_replacements":["4"]}"#,
            0,
        );
        let judge_error = run_generation_suite(
            &[generation_case()],
            Arc::new(LlmBackend::new("candidate").with_executable(candidate_path)),
            Some(Arc::new(LlmBackend::new("judge").with_executable(failed))),
            GenerationEvalConfig::default(),
        )
        .await
        .unwrap_err();
        assert!(judge_error.to_string().contains("zero usable judge"));
    }

    #[test]
    fn reports_use_plain_json_without_hidden_case_fields() {
        let value: Value = serde_json::to_value(GenerationCallReport {
            case_id: "safe-id".into(),
            repeat: 1,
            generation: GenerationOutcome::Error,
            judge: JudgeOutcome::NotRun,
            generation_ms: 1,
            judge_ms: None,
            error: Some("backend unavailable".into()),
            sample: None,
        })
        .unwrap();
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            [
                "case_id",
                "error",
                "generation",
                "generation_ms",
                "judge",
                "judge_ms",
                "repeat"
            ]
        );
    }
}
