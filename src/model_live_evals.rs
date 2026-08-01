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

//! Opt-in live evaluator benchmark.
//!
//! Normal test runs only validate the committed fixture. The ignored test
//! makes paid API calls and must be requested explicitly with:
//!
//! `cargo test model::live_evals::model_effort_benchmark -- --ignored --nocapture`

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use super::GeneratedInstance;
use super::LUNA_MODEL;
use super::PROTOCOL_VERSION;
use super::Verdict;
use super::evaluation_prompt;
use super::evaluation_schema;
use super::parse_evaluation_output;

const CASES_JSON: &str = include_str!("../evals/evaluation_cases.json");
const EFFORTS: [&str; 5] = ["none", "low", "medium", "high", "xhigh"];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalCase {
    id: String,
    tags: Vec<String>,
    question: String,
    criteria: String,
    response: String,
    allowed_verdicts: Vec<Verdict>,
}

#[derive(Debug)]
struct CallResult {
    model: String,
    effort: String,
    case_id: String,
    domain: String,
    allowed_verdicts: Vec<Verdict>,
    elapsed_ms: u128,
    verdict: Option<Verdict>,
    feedback: Option<String>,
    raw_fixture_output: Option<String>,
    output_tokens: Option<u64>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct EffortSummary<'a> {
    model: &'a str,
    effort: &'a str,
    calls: usize,
    usable_calls: usize,
    usable_percent: f64,
    decision_correct: usize,
    decision_accuracy_percent: f64,
    decision_success_percent: f64,
    exact_correct: usize,
    exact_accuracy_percent: f64,
    false_passes: usize,
    false_nays: usize,
    errors: usize,
    mean_ms: u128,
    p50_ms: u128,
    p95_ms: u128,
    max_ms: u128,
    mean_feedback_chars: f64,
    mean_output_tokens: f64,
    by_domain: Vec<DomainSummary>,
}

#[derive(Debug, Serialize)]
struct DomainSummary {
    domain: String,
    calls: usize,
    decision_correct: usize,
    decision_accuracy_percent: f64,
    exact_correct: usize,
    exact_accuracy_percent: f64,
    false_passes: usize,
    false_nays: usize,
}

#[test]
fn evaluation_fixture_is_well_formed() {
    let cases = cases();
    assert!(cases.len() >= 60, "the evaluator suite is too small");
    let expected_domains: HashSet<&str> = [
        "math",
        "physics",
        "chemistry",
        "biology",
        "programming",
        "history",
        "literature",
        "visual-art",
        "language",
        "practical",
    ]
    .into_iter()
    .collect();
    let mut ids = HashSet::new();
    let mut pass = 0;
    let mut negative = 0;
    let mut domain_calls: HashMap<&str, usize> = HashMap::new();
    let mut domain_passes: HashMap<&str, usize> = HashMap::new();
    let mut domain_negatives: HashMap<&str, usize> = HashMap::new();
    let mut represented_verdicts = HashSet::new();
    for case in &cases {
        assert!(!case.id.trim().is_empty());
        assert!(
            ids.insert(case.id.clone()),
            "duplicate case ID: {}",
            case.id
        );
        assert!(!case.tags.is_empty(), "{} has no tags", case.id);
        assert!(
            !case.question.trim().is_empty(),
            "{} has no question",
            case.id
        );
        assert!(
            !case.criteria.trim().is_empty(),
            "{} has no criteria",
            case.id
        );
        assert!(
            !case.allowed_verdicts.is_empty(),
            "{} has no allowed verdicts",
            case.id
        );
        let domain = case.tags.first().expect("tags are nonempty").as_str();
        assert!(
            expected_domains.contains(domain),
            "{} has unexpected primary domain {domain}",
            case.id
        );
        *domain_calls.entry(domain).or_default() += 1;
        represented_verdicts.extend(case.allowed_verdicts.iter().copied());
        let allows_pass = case.allowed_verdicts.contains(&Verdict::Pass);
        assert!(
            !allows_pass || case.allowed_verdicts.len() == 1,
            "{} mixes YEA and NAY labels",
            case.id
        );
        if allows_pass {
            pass += 1;
            *domain_passes.entry(domain).or_default() += 1;
        } else {
            negative += 1;
            *domain_negatives.entry(domain).or_default() += 1;
        }
    }
    assert!(pass >= 20, "the suite needs more positive cases");
    assert!(negative >= 30, "the suite needs more negative cases");
    for domain in expected_domains {
        assert!(
            domain_calls.get(domain).copied().unwrap_or(0) >= 6,
            "{domain} needs at least six cases"
        );
        assert!(
            domain_passes.get(domain).copied().unwrap_or(0) >= 2,
            "{domain} needs at least two YEA cases"
        );
        assert!(
            domain_negatives.get(domain).copied().unwrap_or(0) >= 2,
            "{domain} needs at least two NAY cases"
        );
    }
    for verdict in [
        Verdict::Pass,
        Verdict::Partial,
        Verdict::Fail,
        Verdict::Uncertain,
        Verdict::Invalid,
    ] {
        assert!(
            represented_verdicts.contains(&verdict),
            "suite does not exercise {verdict}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
#[ignore = "paid live benchmark; requires OPENAI_API_KEY"]
async fn model_effort_benchmark() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be present for the paid live benchmark");
    let cases = selected_cases();
    let models = selected_models();
    let efforts = selected_efforts();
    let repeats = env_usize("HASHDRILLS_EVAL_REPEATS", 1).clamp(1, 10);
    let concurrency = env_usize("HASHDRILLS_EVAL_CONCURRENCY", 4).clamp(1, 16);
    let call_count = cases.len() * models.len() * efforts.len() * repeats;
    eprintln!(
        "Hashdrills live eval: {call_count} calls ({} cases × {} models × {} efforts × {repeats})",
        cases.len(),
        models.len(),
        efforts.len()
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .expect("could not build HTTP client");
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut all_results = Vec::with_capacity(call_count);

    for model in &models {
        for effort in &efforts {
            let mut tasks = JoinSet::new();
            let effort_call_count = cases.len() * repeats;
            for repeat in 0..repeats {
                for case in cases.iter().cloned() {
                    let client = client.clone();
                    let api_key = api_key.clone();
                    let semaphore = Arc::clone(&semaphore);
                    let model = model.clone();
                    let effort = effort.to_string();
                    tasks.spawn(async move {
                        let _permit = semaphore.acquire_owned().await.expect("semaphore closed");
                        run_case(client, &api_key, model, effort, case, repeat).await
                    });
                }
            }
            let mut completed = 0;
            while let Some(result) = tasks.join_next().await {
                all_results.push(result.expect("live eval worker panicked"));
                completed += 1;
                if completed % 10 == 0 || completed == effort_call_count {
                    eprintln!(
                        "EVAL_PROGRESS model={model} effort={effort} completed={completed}/{effort_call_count}"
                    );
                }
            }

            let effort_results: Vec<&CallResult> = all_results
                .iter()
                .filter(|result| result.model == *model && result.effort == *effort)
                .collect();
            print_summary(model, effort, &effort_results);
            print_misses(&effort_results);
            if std::env::var_os("HASHDRILLS_EVAL_SHOW_FEEDBACK").is_some() {
                print_feedback(&effort_results);
            }
        }
    }

    assert!(
        all_results.iter().any(|result| result.verdict.is_some()),
        "every live model call failed"
    );
}

fn selected_models() -> Vec<String> {
    let Ok(selected) = std::env::var("HASHDRILLS_EVAL_MODELS") else {
        return vec![LUNA_MODEL.into()];
    };
    let requested: Vec<String> = selected
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect();
    assert!(!requested.is_empty(), "no evaluation models selected");
    requested
}

fn supports_verbosity(model: &str) -> bool {
    model.starts_with("gpt-5") && model != LUNA_MODEL
}

fn cases() -> Vec<EvalCase> {
    serde_json::from_str(CASES_JSON).expect("evaluation fixture must be valid JSON")
}

fn selected_cases() -> Vec<EvalCase> {
    let all_cases = cases();
    let Ok(selected) = std::env::var("HASHDRILLS_EVAL_CASE_IDS") else {
        return all_cases;
    };
    let requested: HashSet<&str> = selected
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    assert!(!requested.is_empty(), "no evaluation cases selected");
    let selected_cases: Vec<EvalCase> = all_cases
        .into_iter()
        .filter(|case| requested.contains(case.id.as_str()))
        .collect();
    assert_eq!(
        selected_cases.len(),
        requested.len(),
        "one or more requested evaluation case IDs do not exist"
    );
    selected_cases
}

fn selected_efforts() -> Vec<&'static str> {
    let Some(selected) = std::env::var("HASHDRILLS_EVAL_EFFORTS").ok() else {
        return EFFORTS.to_vec();
    };
    let requested: Vec<&str> = selected
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    assert!(!requested.is_empty(), "no evaluation efforts selected");
    requested
        .into_iter()
        .map(|requested| {
            EFFORTS
                .into_iter()
                .find(|effort| *effort == requested)
                .unwrap_or_else(|| panic!("unsupported evaluation effort: {requested}"))
        })
        .collect()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

async fn run_case(
    client: reqwest::Client,
    api_key: &str,
    model: String,
    effort: String,
    case: EvalCase,
    _repeat: usize,
) -> CallResult {
    let instance = GeneratedInstance {
        question: case.question,
        target: case.criteria.clone(),
        rubric: case.criteria,
        model: "eval-fixture".into(),
        protocol_version: PROTOCOL_VERSION,
    };
    let prompt = evaluation_prompt(&instance, &case.response).expect("fixture prompt is valid");
    let schema: Value =
        serde_json::from_str(&evaluation_schema()).expect("evaluation schema is valid");
    let mut request = json!({
        "model": &model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": schema}
        }
    });
    if supports_verbosity(&model) {
        request["verbosity"] = json!("low");
    }
    // Measure wall time to a usable production judgment, including response
    // body download, JSON decoding, and `parse_evaluation_output`.
    let started = Instant::now();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .await;

    let mut result = CallResult {
        model,
        effort,
        case_id: case.id,
        domain: case
            .tags
            .first()
            .cloned()
            .unwrap_or_else(|| "untagged".into()),
        allowed_verdicts: case.allowed_verdicts,
        elapsed_ms: 0,
        verdict: None,
        feedback: None,
        raw_fixture_output: None,
        output_tokens: None,
        error: None,
    };
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            result.error = Some(format!("transport:{:?}", error.status()));
            result.elapsed_ms = started.elapsed().as_millis();
            return result;
        }
    };
    let status = response.status();
    let body: Value = match response.json().await {
        Ok(body) => body,
        Err(_) => {
            result.error = Some(format!("http:{}:invalid-response-json", status.as_u16()));
            result.elapsed_ms = started.elapsed().as_millis();
            return result;
        }
    };
    if !status.is_success() {
        let error_type = body
            .pointer("/error/type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let error_code = body
            .pointer("/error/code")
            .and_then(Value::as_str)
            .unwrap_or("none");
        let error_param = body
            .pointer("/error/param")
            .and_then(Value::as_str)
            .unwrap_or("none");
        result.error = Some(format!(
            "http:{}:type={error_type}:code={error_code}:param={error_param}",
            status.as_u16()
        ));
        result.elapsed_ms = started.elapsed().as_millis();
        return result;
    }
    result.output_tokens = body
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64);
    let Some(content) = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    else {
        result.error = Some("missing-content".into());
        result.elapsed_ms = started.elapsed().as_millis();
        return result;
    };
    match parse_evaluation_output(content.as_bytes()) {
        Ok(output) => {
            result.verdict = Some(output.verdict);
            result.feedback = Some(output.feedback);
        }
        Err(error) => {
            result.error = Some(format!("protocol-parse:{error}"));
            // This benchmark uses only the committed public fixture. Product
            // model responses remain opaque and are never captured this way.
            result.raw_fixture_output = Some(content.into());
        }
    }
    result.elapsed_ms = started.elapsed().as_millis();
    result
}

fn print_summary<'a>(model: &'a str, effort: &'a str, results: &[&CallResult]) {
    let successful: Vec<&CallResult> = results
        .iter()
        .copied()
        .filter(|result| result.verdict.is_some())
        .collect();
    let decision_correct = successful
        .iter()
        .filter(|result| decision_is_correct(result))
        .count();
    let exact_correct = successful
        .iter()
        .filter(|result| {
            result
                .allowed_verdicts
                .contains(&result.verdict.expect("successful result"))
        })
        .count();
    let false_passes = successful
        .iter()
        .filter(|result| {
            result.verdict == Some(Verdict::Pass)
                && !result.allowed_verdicts.contains(&Verdict::Pass)
        })
        .count();
    let false_nays = successful
        .iter()
        .filter(|result| {
            result.verdict != Some(Verdict::Pass)
                && result.allowed_verdicts.contains(&Verdict::Pass)
        })
        .count();
    let mut latencies: Vec<u128> = successful.iter().map(|result| result.elapsed_ms).collect();
    latencies.sort_unstable();
    let feedback_chars: usize = successful
        .iter()
        .filter_map(|result| result.feedback.as_ref())
        .map(|feedback| feedback.chars().count())
        .sum();
    let output_tokens: u64 = successful
        .iter()
        .filter_map(|result| result.output_tokens)
        .sum();
    let summary = EffortSummary {
        model,
        effort,
        calls: results.len(),
        usable_calls: successful.len(),
        usable_percent: percent(successful.len(), results.len()),
        decision_correct,
        decision_accuracy_percent: percent(decision_correct, successful.len()),
        decision_success_percent: percent(decision_correct, results.len()),
        exact_correct,
        exact_accuracy_percent: percent(exact_correct, successful.len()),
        false_passes,
        false_nays,
        errors: results.len().saturating_sub(successful.len()),
        mean_ms: mean_u128(&latencies),
        p50_ms: percentile(&latencies, 50),
        p95_ms: percentile(&latencies, 95),
        max_ms: latencies.last().copied().unwrap_or(0),
        mean_feedback_chars: average(feedback_chars as u64, successful.len()),
        mean_output_tokens: average(output_tokens, successful.len()),
        by_domain: domain_summaries(results),
    };
    eprintln!(
        "EVAL_SUMMARY {}",
        serde_json::to_string(&summary).expect("summary serializes")
    );
}

fn print_misses(results: &[&CallResult]) {
    for result in results {
        if decision_is_correct(result) {
            continue;
        }
        eprintln!(
            "EVAL_MISS model={} effort={} case={} expected={:?} actual={:?} feedback={:?} error={:?}",
            result.model,
            result.effort,
            result.case_id,
            result.allowed_verdicts,
            result.verdict,
            result.feedback,
            result.error
        );
    }
}

fn print_feedback(results: &[&CallResult]) {
    for result in results {
        if let Some(feedback) = &result.feedback {
            eprintln!(
                "EVAL_FEEDBACK model={} effort={} case={} verdict={:?} chars={} feedback={feedback:?}",
                result.model,
                result.effort,
                result.case_id,
                result.verdict,
                feedback.chars().count()
            );
        } else if let Some(raw_output) = &result.raw_fixture_output {
            eprintln!(
                "EVAL_RAW_FIXTURE_OUTPUT model={} effort={} case={} output={raw_output:?}",
                result.model, result.effort, result.case_id
            );
        }
    }
}

fn decision_is_correct(result: &CallResult) -> bool {
    result.verdict.is_some_and(|verdict| {
        (verdict == Verdict::Pass) == result.allowed_verdicts.contains(&Verdict::Pass)
    })
}

fn domain_summaries(results: &[&CallResult]) -> Vec<DomainSummary> {
    let mut domains: Vec<&str> = results
        .iter()
        .map(|result| result.domain.as_str())
        .collect();
    domains.sort_unstable();
    domains.dedup();
    domains
        .into_iter()
        .map(|domain| {
            let domain_results: Vec<&CallResult> = results
                .iter()
                .copied()
                .filter(|result| result.domain == domain)
                .collect();
            let successful: Vec<&CallResult> = domain_results
                .iter()
                .copied()
                .filter(|result| result.verdict.is_some())
                .collect();
            let decision_correct = successful
                .iter()
                .filter(|result| decision_is_correct(result))
                .count();
            let exact_correct = successful
                .iter()
                .filter(|result| {
                    result
                        .verdict
                        .is_some_and(|verdict| result.allowed_verdicts.contains(&verdict))
                })
                .count();
            let false_passes = successful
                .iter()
                .filter(|result| {
                    result.verdict == Some(Verdict::Pass)
                        && !result.allowed_verdicts.contains(&Verdict::Pass)
                })
                .count();
            let false_nays = successful
                .iter()
                .filter(|result| {
                    result.verdict != Some(Verdict::Pass)
                        && result.allowed_verdicts.contains(&Verdict::Pass)
                })
                .count();
            DomainSummary {
                domain: domain.into(),
                calls: domain_results.len(),
                decision_correct,
                decision_accuracy_percent: percent(decision_correct, successful.len()),
                exact_correct,
                exact_accuracy_percent: percent(exact_correct, successful.len()),
                false_passes,
                false_nays,
            }
        })
        .collect()
}

fn percentile(sorted: &[u128], percentile: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let index = (sorted.len() * percentile).div_ceil(100).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn mean_u128(values: &[u128]) -> u128 {
    if values.is_empty() {
        0
    } else {
        values.iter().sum::<u128>() / values.len() as u128
    }
}

fn average(total: u64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn percent(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}
