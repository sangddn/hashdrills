// Copyright 2025–2026 Fernando Borretti
// Modified in 2026 for Hashdrills.
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

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::io::Read;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Args;
use clap::Parser;
use clap::ValueEnum;
use serde::Serialize;

use crate::archive::prepare_archive_directory;
use crate::error::ErrorReport;
use crate::error::Fallible;
use crate::error::fail;
use crate::evals::EvalError;
use crate::evals::EvalRunConfig;
use crate::evals::EvaluationCallReport;
use crate::evals::EvaluationReport;
use crate::evals::GenerationCallReport;
use crate::evals::GenerationEvalConfig;
use crate::evals::GenerationReport;
use crate::evals::JudgeOutcome;
use crate::evals::bundled_evaluation_cases;
use crate::evals::bundled_generation_cases;
use crate::evals::load_evaluation_cases;
use crate::evals::load_generation_cases;
use crate::evals::run_evaluation_suite;
use crate::evals::run_generation_suite;
use crate::model::DEFAULT_TIMEOUT;
use crate::model::LlmBackend;
use crate::model::LlmOption;
use crate::model::ModelBackend;
use crate::model::PROMPT_DEFAULT_INSTRUCTIONS_PLACEHOLDER;
use crate::model::PROMPT_INPUT_JSON_PLACEHOLDER;
use crate::model::ReasoningEffort;
use crate::model::SchemaMode;
use crate::selection::Selection;
use crate::selection::select_specs;
use crate::spec::DrillSpec;
use crate::spec::parse_path;
use crate::storage::Storage;
use crate::storage::StorageStats;
use crate::types::date::Date;
use crate::types::performance::Performance;
use crate::web::ServerConfig;
use crate::web::start_server;

const DEFAULT_MODEL: &str = "gpt-5.5-2026-04-23";
const DEFAULT_MODEL_REASONING_EFFORT: ReasoningEffort = ReasoningEffort::None;
const DEFAULT_JUDGE_MODEL: &str = "gpt-5.6-sol";
const MAX_CUSTOMIZATION_FILE_BYTES: u64 = 64 * 1024;

/// Composable collection selectors shared by every command that consumes
/// drills. Includes form a union; excludes are applied after that union.
#[derive(Args, Clone, Debug, Default, Eq, PartialEq)]
struct SelectionArgs {
    /// Include this exact logical deck. Repeat to select a union of decks.
    #[arg(long, visible_alias = "from-deck")]
    include_deck: Vec<String>,
    /// Include a source file, or every deck below a directory. Repeatable.
    #[arg(long)]
    include_path: Vec<PathBuf>,
    /// Exclude this exact logical deck. Exclusions always win. Repeatable.
    #[arg(long)]
    exclude_deck: Vec<String>,
    /// Exclude a source file, or every deck below a directory. Repeatable.
    #[arg(long)]
    exclude_path: Vec<PathBuf>,
}

impl From<SelectionArgs> for Selection {
    fn from(value: SelectionArgs) -> Self {
        Self {
            include_decks: value.include_deck,
            include_paths: value.include_path,
            exclude_decks: value.exclude_deck,
            exclude_paths: value.exclude_path,
        }
    }
}

/// Provider-neutral configuration for Simon Willison's `llm` CLI.
///
/// Provider-specific behavior belongs in repeatable `KEY=VALUE` options, so
/// installed OpenAI, OpenRouter, Ollama, and local-provider plugins remain the
/// authority for their own settings.
#[derive(Args, Clone, Debug)]
struct CommonLlmArgs {
    /// Fallback model for generation and evaluation; the built-in default uses `none` reasoning.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,
    /// Provider option passed as process arguments to both stages; never put credentials here.
    #[arg(long = "llm-option", value_name = "KEY=VALUE")]
    llm_option: Vec<LlmOption>,
    /// Send the JSON schema natively, or include it in the prompt.
    #[arg(long, default_value_t = SchemaMode::default())]
    schema_mode: SchemaMode,
    /// Override the safely PATH-resolved `llm` executable. The selected path runs as local code.
    #[arg(long)]
    llm_executable: Option<OsString>,
    /// Maximum duration of each model request, in seconds.
    #[arg(
        long,
        default_value_t = DEFAULT_TIMEOUT.as_secs(),
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    llm_timeout: u64,
}

#[derive(Args, Clone, Debug, Default)]
struct GenerationLlmArgs {
    /// Model used only to generate concrete drill instances.
    #[arg(long)]
    generation_model: Option<String>,
    /// Override the reasoning effort used only for generation.
    #[arg(long)]
    generation_reasoning_effort: Option<ReasoningEffort>,
    /// Provider option passed as process arguments only for generation.
    #[arg(long = "generation-llm-option", value_name = "KEY=VALUE")]
    generation_llm_option: Vec<LlmOption>,
    /// Extra trusted instructions appended to the generation prompt.
    #[arg(long, conflicts_with = "generation_instructions_file")]
    generation_instructions: Option<String>,
    /// Read extra trusted generation instructions from this UTF-8 file.
    #[arg(long, value_name = "FILE", conflicts_with = "generation_instructions")]
    generation_instructions_file: Option<PathBuf>,
    /// Advanced generation prompt template containing `{{input_json}}`.
    #[arg(long, value_name = "FILE")]
    generation_prompt_template_file: Option<PathBuf>,
}

#[derive(Args, Clone, Debug, Default)]
struct EvaluationLlmArgs {
    /// Model used only to evaluate submitted answers.
    #[arg(long)]
    evaluation_model: Option<String>,
    /// Override the reasoning effort used only for evaluation.
    #[arg(long)]
    evaluation_reasoning_effort: Option<ReasoningEffort>,
    /// Provider option passed as process arguments only for evaluation.
    #[arg(long = "evaluation-llm-option", value_name = "KEY=VALUE")]
    evaluation_llm_option: Vec<LlmOption>,
    /// Extra trusted instructions appended to the evaluation prompt.
    #[arg(long, conflicts_with = "evaluation_instructions_file")]
    evaluation_instructions: Option<String>,
    /// Read extra trusted evaluation instructions from this UTF-8 file.
    #[arg(long, value_name = "FILE", conflicts_with = "evaluation_instructions")]
    evaluation_instructions_file: Option<PathBuf>,
    /// Advanced evaluation prompt template containing `{{input_json}}`.
    #[arg(long, value_name = "FILE")]
    evaluation_prompt_template_file: Option<PathBuf>,
}

#[derive(Args, Clone, Debug)]
struct LlmArgs {
    #[command(flatten)]
    common: CommonLlmArgs,
    #[command(flatten)]
    generation: GenerationLlmArgs,
    #[command(flatten)]
    evaluation: EvaluationLlmArgs,
}

#[derive(Args, Clone, Debug)]
struct JudgeLlmArgs {
    /// Model used by the blind generation-quality judge.
    #[arg(long, default_value = DEFAULT_JUDGE_MODEL)]
    judge_model: String,
    /// Reasoning effort used by the blind judge.
    #[arg(long, default_value_t = ReasoningEffort::High)]
    judge_reasoning_effort: ReasoningEffort,
    /// Provider option passed only to the blind judge as KEY=VALUE.
    #[arg(long = "judge-llm-option", value_name = "KEY=VALUE")]
    judge_llm_option: Vec<LlmOption>,
    /// Maximum simultaneous blind-judge calls.
    #[arg(long, default_value_t = 1, value_parser = parse_eval_concurrency)]
    judge_concurrency: usize,
}

/// Controls shared by answer and generation evaluation suites.
#[derive(Args, Clone, Debug)]
struct EvalRunArgs {
    /// Calls per selected case.
    #[arg(long, default_value_t = 1, value_parser = parse_eval_repeats)]
    repeats: usize,
    /// Maximum simultaneous candidate calls.
    #[arg(long, default_value_t = 1, value_parser = parse_eval_concurrency)]
    concurrency: usize,
    /// Run this exact case ID. Repeat to select multiple cases.
    #[arg(long = "case", value_name = "ID")]
    case_ids: Vec<String>,
    /// Deterministic seed for generated variation keys.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Report format. JSON never includes provider option values or prompt bodies.
    #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
    format: OutputFormat,
}

impl EvalRunArgs {
    fn to_config(&self, call_timeout: Duration) -> EvalRunConfig {
        EvalRunConfig {
            repeats: self.repeats,
            concurrency: self.concurrency,
            case_ids: self.case_ids.clone(),
            seed: self.seed,
            call_timeout,
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
// Clap owns this short-lived parse tree; boxing fields would complicate its
// derive surface without reducing steady-state memory.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Validate authored drill specifications without opening the database.
    Check {
        /// Markdown file or collection directory. Defaults to the current directory.
        path: Option<PathBuf>,
    },
    /// Generate previews without enrolling drills or opening the database.
    Sample {
        /// Markdown file or collection directory. Defaults to the current directory.
        path: Option<PathBuf>,
        /// Number of previews to generate for each matching drill specification.
        #[arg(long, default_value_t = 3)]
        count: usize,
        #[command(flatten)]
        common: CommonLlmArgs,
        #[command(flatten)]
        generation: GenerationLlmArgs,
        #[command(flatten)]
        selection: SelectionArgs,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
    },
    /// Practice due and new drills in a local web interface.
    Drill {
        /// Collection directory. Defaults to the current directory.
        path: Option<PathBuf>,
        #[command(flatten)]
        llm: LlmArgs,
        /// Maximum number of unseen drills to introduce in this session.
        #[arg(long, visible_alias = "new-card-limit")]
        new_drill_limit: Option<usize>,
        /// Maximum total number of drills in this session.
        #[arg(long, visible_alias = "card-limit")]
        drill_limit: Option<usize>,
        #[command(flatten)]
        selection: SelectionArgs,
        /// Archive each resolved generated drill below this directory.
        #[arg(long, value_name = "DIR")]
        save_generated: Option<PathBuf>,
        /// Address on which to bind the web server.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port on which to bind the web server.
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// Open the practice interface in the default browser.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        open_browser: bool,
        /// Public base URL shown for remote access and generated links.
        #[arg(long, value_name = "URL")]
        public_url: Option<String>,
        /// Disable session authentication (restricted to loopback by the server).
        #[arg(long)]
        no_auth: bool,
    },
    /// Print logical decks with drills due on a date.
    Due {
        /// Collection directory. Defaults to the current directory.
        path: Option<PathBuf>,
        /// Date to inspect: `today`, `tomorrow`, or YYYY-MM-DD.
        #[arg(default_value = "today")]
        date: String,
        #[command(flatten)]
        selection: SelectionArgs,
    },
    /// Print scheduler and attempt statistics.
    Stats {
        /// Collection directory. Defaults to the current directory.
        path: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
        #[command(flatten)]
        selection: SelectionArgs,
    },
    /// Benchmark answer judgment or generated drill quality without opening a database.
    Eval {
        #[command(subcommand)]
        suite: EvalCommand,
    },
}

#[derive(clap::Subcommand, Debug)]
enum EvalCommand {
    /// Benchmark answer verdicts against labelled cases.
    Answers {
        /// Strict JSON case array. Omit to use the bundled suite.
        cases: Option<PathBuf>,
        #[command(flatten)]
        common: CommonLlmArgs,
        #[command(flatten)]
        evaluation: EvaluationLlmArgs,
        #[command(flatten)]
        run: EvalRunArgs,
    },
    /// Benchmark generated drills, optionally with a blind quality judge.
    Generation {
        /// Strict JSON case array. Omit to use the bundled suite.
        cases: Option<PathBuf>,
        #[command(flatten)]
        common: CommonLlmArgs,
        #[command(flatten)]
        generation: GenerationLlmArgs,
        #[command(flatten)]
        run: EvalRunArgs,
        #[command(flatten)]
        judge: JudgeLlmArgs,
        /// Generate and measure diversity without running the quality judge.
        #[arg(long)]
        skip_judge: bool,
        /// Include generated Q/A/rubric text in output; may expose private material.
        #[arg(long)]
        show_samples: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    #[default]
    Human,
    Json,
}

#[derive(Serialize)]
struct SampleRecord {
    deck: String,
    spec_hash: String,
    sample: usize,
    question: String,
    target: String,
    rubric: String,
    model: String,
    protocol_version: u32,
}

pub async fn entrypoint() -> Fallible<()> {
    run(Cli::parse()).await
}

async fn run(cli: Cli) -> Fallible<()> {
    match cli.command {
        Command::Check { path } => check(path),
        Command::Sample {
            path,
            count,
            common,
            generation,
            selection,
            format,
        } => sample(path, count, common, generation, selection, format).await,
        Command::Drill {
            path,
            llm,
            new_drill_limit,
            drill_limit,
            selection,
            save_generated,
            host,
            port,
            open_browser,
            public_url,
            no_auth,
        } => {
            drill(
                path,
                llm,
                new_drill_limit,
                drill_limit,
                selection,
                save_generated,
                host,
                port,
                open_browser,
                public_url,
                no_auth,
            )
            .await
        }
        Command::Due {
            path,
            date,
            selection,
        } => due(path, parse_date(&date)?, selection),
        Command::Stats {
            path,
            format,
            selection,
        } => stats(path, format, selection),
        Command::Eval { suite } => eval(suite).await,
    }
}

async fn eval(command: EvalCommand) -> Fallible<()> {
    match command {
        EvalCommand::Answers {
            cases,
            common,
            evaluation,
            run,
        } => eval_answers(cases, common, evaluation, run).await,
        EvalCommand::Generation {
            cases,
            common,
            generation,
            run,
            judge,
            skip_judge,
            show_samples,
        } => {
            eval_generation(
                cases,
                common,
                generation,
                run,
                judge,
                skip_judge,
                show_samples,
            )
            .await
        }
    }
}

async fn eval_answers(
    cases_path: Option<PathBuf>,
    common: CommonLlmArgs,
    evaluation: EvaluationLlmArgs,
    run: EvalRunArgs,
) -> Fallible<()> {
    // Strict case loading precedes backend construction, and this command
    // never constructs Storage.
    let cases = match cases_path {
        Some(path) => load_evaluation_cases(&path),
        None => bundled_evaluation_cases(),
    }
    .map_err(eval_error)?;
    validate_case_selection(&run.case_ids, cases.iter().map(|case| case.id.as_str()))?;
    let call_timeout = Duration::from_secs(common.llm_timeout);
    let backend: Arc<dyn ModelBackend> =
        Arc::new(build_scoped_backend(common, None, Some(evaluation))?);
    let report = run_evaluation_suite(&cases, backend, run.to_config(call_timeout))
        .await
        .map_err(eval_error)?;
    match run.format {
        OutputFormat::Human => print_evaluation_report(&report),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }
    Ok(())
}

async fn eval_generation(
    cases_path: Option<PathBuf>,
    common: CommonLlmArgs,
    generation: GenerationLlmArgs,
    run: EvalRunArgs,
    judge_args: JudgeLlmArgs,
    skip_judge: bool,
    show_samples: bool,
) -> Fallible<()> {
    let cases = match cases_path {
        Some(path) => load_generation_cases(&path),
        None => bundled_generation_cases(),
    }
    .map_err(eval_error)?;
    validate_case_selection(&run.case_ids, cases.iter().map(|case| case.id.as_str()))?;
    let call_timeout = Duration::from_secs(common.llm_timeout);
    let judge_concurrency = judge_args.judge_concurrency;
    let judge = if skip_judge {
        None
    } else {
        Some(Arc::new(build_judge_backend(&common, judge_args)?))
    };
    let candidate = Arc::new(build_scoped_backend(common, Some(generation), None)?);
    let report = run_generation_suite(
        &cases,
        candidate,
        judge,
        GenerationEvalConfig {
            run: run.to_config(call_timeout),
            judge_concurrency,
            skip_judge,
            include_samples: show_samples,
        },
    )
    .await
    .map_err(eval_error)?;
    match run.format {
        OutputFormat::Human => print_generation_report(&report, show_samples),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }
    Ok(())
}

fn eval_error(error: EvalError) -> ErrorReport {
    ErrorReport::new(format!("eval: {error}"))
}

fn validate_case_selection<'a>(
    requested: &[String],
    available: impl Iterator<Item = &'a str>,
) -> Fallible<()> {
    if requested.is_empty() {
        return Ok(());
    }
    let available: HashSet<&str> = available.collect();
    let mut seen = HashSet::new();
    for case_id in requested {
        if !seen.insert(case_id.as_str()) {
            return fail(format!(
                "eval: case ID '{case_id}' was selected more than once"
            ));
        }
        if !available.contains(case_id.as_str()) {
            return fail(format!("eval: unknown case ID '{case_id}'"));
        }
    }
    Ok(())
}

/// Parse and validate only. In particular, this function must never construct
/// a [`Storage`] or a model backend.
fn check(path: Option<PathBuf>) -> Fallible<()> {
    let path = resolve_path(path)?;
    let specs = parse_specs(&path)?;
    println!("checked {} drill specification(s)", specs.len());
    Ok(())
}

/// Generate authoring previews without constructing [`Storage`]. This is an
/// intentional boundary: sampling must not enroll specs or create a database.
async fn sample(
    path: Option<PathBuf>,
    count: usize,
    common: CommonLlmArgs,
    generation: GenerationLlmArgs,
    selection: SelectionArgs,
    format: OutputFormat,
) -> Fallible<()> {
    if count == 0 {
        return fail("sample count must be at least 1");
    }

    let path = resolve_path(path)?;
    let specs = parse_specs(&path)?;
    // Selection is resolved before constructing a backend. A typo must never
    // reach a provider or create collection state.
    let specs = selected_specs(&specs, &path, selection, true)?;
    let backend = build_scoped_backend(common, Some(generation), None)?;
    let mut records = Vec::with_capacity(specs.len().saturating_mul(count));

    for spec in &specs {
        for sample_index in 1..=count {
            let generated = backend.generate(spec).await?;
            records.push(SampleRecord {
                deck: spec.deck_name.clone(),
                spec_hash: spec.hash().to_string(),
                sample: sample_index,
                question: generated.question,
                target: generated.target,
                rubric: generated.rubric,
                model: generated.model,
                protocol_version: generated.protocol_version,
            });
        }
    }

    match format {
        OutputFormat::Human => print_samples_human(&records),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&records)?),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn drill(
    path: Option<PathBuf>,
    llm: LlmArgs,
    new_drill_limit: Option<usize>,
    drill_limit: Option<usize>,
    selection: SelectionArgs,
    save_generated: Option<PathBuf>,
    host: String,
    port: u16,
    open_browser: bool,
    public_url: Option<String>,
    no_auth: bool,
) -> Fallible<()> {
    let path = resolve_path(path)?;
    require_collection_directory(&path, "drill")?;
    let specs = parse_specs(&path)?;
    // Finish selector and model-configuration validation before opening the
    // database or binding a server.
    let specs = selected_specs(&specs, &path, selection, true)?;
    let root = collection_root(&path)?;
    let canonical_root = root.canonicalize()?;
    let save_generated = validate_archive_directory(save_generated, &canonical_root)?
        .map(prepare_archive_directory)
        .transpose()?;
    let backend = build_backend(llm)?;
    let storage = Storage::open(root)?;

    let config = ServerConfig {
        specs,
        storage,
        collection_root: canonical_root,
        backend: Arc::new(backend),
        host,
        port,
        deck_filter: None,
        due_limit: drill_limit,
        new_limit: new_drill_limit,
        archive_dir: save_generated,
        public_url,
        open_browser,
        auth_enabled: !no_auth,
    };
    start_server(config).await
}

fn due(path: Option<PathBuf>, date: Date, selection: SelectionArgs) -> Fallible<()> {
    let path = resolve_path(path)?;
    require_collection_directory(&path, "due")?;
    let specs = parse_specs(&path)?;
    let specs = selected_specs(&specs, &path, selection, false)?;
    let root = collection_root(&path)?;
    let storage = Storage::open(root)?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for spec in specs {
        let is_due_on_date = matches!(
            storage.performance_opt(spec.hash())?,
            Some(Performance::Reviewed(performance)) if performance.due_date == date
        );
        if is_due_on_date {
            *counts.entry(spec.deck_name).or_default() += 1;
        }
    }
    if counts.is_empty() {
        println!("No drills due on {date}.");
    } else {
        let total: usize = counts.values().sum();
        for (deck, count) in counts {
            println!("{}\t{count}", terminal_safe(&deck));
        }
        println!("Total\t{total}");
    }
    Ok(())
}

fn stats(path: Option<PathBuf>, format: OutputFormat, selection: SelectionArgs) -> Fallible<()> {
    let path = resolve_path(path)?;
    require_collection_directory(&path, "stats")?;
    let specs = parse_specs(&path)?;
    let specs = selected_specs(&specs, &path, selection, false)?;
    let root = collection_root(&path)?;
    let storage = Storage::open(root)?;
    let stats = storage.stats_for_specs(Date::today(), &specs)?;
    match format {
        OutputFormat::Human => print_stats_human(stats),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&stats)?),
    }
    Ok(())
}

fn require_collection_directory(path: &Path, command: &str) -> Fallible<()> {
    let metadata = fs::metadata(path).map_err(|error| {
        ErrorReport::new(format!(
            "could not inspect {command} collection path '{}': {error}",
            path.display()
        ))
    })?;
    if !metadata.is_dir() {
        return fail(format!(
            "{command} requires a collection directory, not '{}'; select files with --include-path",
            path.display()
        ));
    }
    Ok(())
}

/// Resolve a possibly not-yet-created archive directory without creating it.
///
/// Canonicalizing the nearest existing ancestor makes symlinks authoritative;
/// normalizing the absent suffix makes `.` and `..` unable to disguise a path
/// below the collection. The archive module remains responsible for securely
/// creating the final outside directory.
fn validate_archive_directory(
    archive_dir: Option<PathBuf>,
    collection_root: &Path,
) -> Fallible<Option<PathBuf>> {
    let Some(authored) = archive_dir else {
        return Ok(None);
    };
    if authored.as_os_str().is_empty() {
        return fail("--save-generated may not be empty");
    }
    let absolute = if authored.is_absolute() {
        authored
    } else {
        std::env::current_dir()?.join(authored)
    };
    let normalized = normalize_absolute_path(&absolute);
    let resolved = resolve_from_existing_ancestor(&normalized)?;
    if resolved == collection_root || resolved.starts_with(collection_root) {
        return fail(format!(
            "--save-generated directory '{}' must be outside collection root '{}'",
            resolved.display(),
            collection_root.display()
        ));
    }
    Ok(Some(resolved))
}

fn normalize_absolute_path(path: &Path) -> PathBuf {
    debug_assert!(path.is_absolute());
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    normalized
}

fn resolve_from_existing_ancestor(path: &Path) -> Fallible<PathBuf> {
    let mut ancestor = path.to_path_buf();
    let mut missing = Vec::<OsString>::new();
    loop {
        match fs::symlink_metadata(&ancestor) {
            Ok(metadata) => {
                if missing.is_empty() && metadata.file_type().is_symlink() {
                    return fail(format!(
                        "--save-generated directory '{}' may not be a symbolic link",
                        path.display()
                    ));
                }
                let canonical = ancestor.canonicalize().map_err(|error| {
                    ErrorReport::new(format!(
                        "could not resolve --save-generated ancestor '{}': {error}",
                        ancestor.display()
                    ))
                })?;
                if !canonical.is_dir() {
                    return fail(format!(
                        "--save-generated ancestor '{}' is not a directory",
                        canonical.display()
                    ));
                }
                let mut resolved = canonical;
                for component in missing.iter().rev() {
                    resolved.push(component);
                }
                return Ok(resolved);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                let component = ancestor.file_name().ok_or_else(|| {
                    ErrorReport::new(format!(
                        "could not find an existing ancestor of --save-generated path '{}'",
                        path.display()
                    ))
                })?;
                missing.push(component.to_os_string());
                if !ancestor.pop() {
                    return fail(format!(
                        "could not find an existing ancestor of --save-generated path '{}'",
                        path.display()
                    ));
                }
            }
            Err(error) => {
                return Err(ErrorReport::new(format!(
                    "could not inspect --save-generated path '{}': {error}",
                    ancestor.display()
                )));
            }
        }
    }
}

fn selected_specs(
    specs: &[DrillSpec],
    collection_path: &Path,
    selection: SelectionArgs,
    require_nonempty: bool,
) -> Fallible<Vec<DrillSpec>> {
    let selection = Selection::from(selection);
    let selected = select_specs(specs, collection_path, &selection)?;
    if require_nonempty && selected.is_empty() {
        return fail("selection matched no drill specifications");
    }
    Ok(selected.into_iter().cloned().collect())
}

fn build_backend(args: LlmArgs) -> Fallible<LlmBackend> {
    build_scoped_backend(args.common, Some(args.generation), Some(args.evaluation))
}

fn build_judge_backend(common: &CommonLlmArgs, judge: JudgeLlmArgs) -> Fallible<LlmBackend> {
    validate_model_argument("--judge-model", &judge.judge_model)?;
    if common
        .llm_option
        .iter()
        .chain(&judge.judge_llm_option)
        .any(|option| option.key() == "reasoning_effort")
    {
        return fail("--judge-reasoning-effort conflicts with a judge reasoning_effort LLM option");
    }
    build_scoped_backend(
        common.clone(),
        None,
        Some(EvaluationLlmArgs {
            evaluation_model: Some(judge.judge_model),
            evaluation_reasoning_effort: Some(judge.judge_reasoning_effort),
            evaluation_llm_option: judge.judge_llm_option,
            ..EvaluationLlmArgs::default()
        }),
    )
}

fn build_scoped_backend(
    common: CommonLlmArgs,
    generation: Option<GenerationLlmArgs>,
    evaluation: Option<EvaluationLlmArgs>,
) -> Fallible<LlmBackend> {
    let generation_enabled = generation.is_some();
    let evaluation_enabled = evaluation.is_some();
    let CommonLlmArgs {
        model,
        llm_option,
        schema_mode,
        llm_executable,
        llm_timeout,
    } = common;
    let GenerationLlmArgs {
        generation_model,
        mut generation_reasoning_effort,
        generation_llm_option,
        generation_instructions,
        generation_instructions_file,
        generation_prompt_template_file,
    } = generation.unwrap_or_default();
    let EvaluationLlmArgs {
        evaluation_model,
        mut evaluation_reasoning_effort,
        evaluation_llm_option,
        evaluation_instructions,
        evaluation_instructions_file,
        evaluation_prompt_template_file,
    } = evaluation.unwrap_or_default();

    validate_model_argument("--model", &model)?;
    if let Some(value) = generation_model.as_deref() {
        validate_model_argument("--generation-model", value)?;
    }
    if let Some(value) = evaluation_model.as_deref() {
        validate_model_argument("--evaluation-model", value)?;
    }
    if llm_executable
        .as_ref()
        .is_some_and(|value| value.is_empty())
    {
        return fail("--llm-executable may not be empty");
    }
    if llm_timeout == 0 {
        return fail("--llm-timeout must be at least 1 second");
    }
    if generation_reasoning_effort.is_some()
        && llm_option
            .iter()
            .chain(&generation_llm_option)
            .any(|option| option.key() == "reasoning_effort")
    {
        return fail(
            "--generation-reasoning-effort conflicts with a generation reasoning_effort LLM option",
        );
    }
    if evaluation_reasoning_effort.is_some()
        && llm_option
            .iter()
            .chain(&evaluation_llm_option)
            .any(|option| option.key() == "reasoning_effort")
    {
        return fail(
            "--evaluation-reasoning-effort conflicts with an evaluation reasoning_effort LLM option",
        );
    }

    generation_reasoning_effort = effective_reasoning_effort(
        generation_enabled,
        generation_model.as_deref().unwrap_or(model.as_str()),
        generation_reasoning_effort,
        &llm_option,
        &generation_llm_option,
    );
    evaluation_reasoning_effort = effective_reasoning_effort(
        evaluation_enabled,
        evaluation_model.as_deref().unwrap_or(model.as_str()),
        evaluation_reasoning_effort,
        &llm_option,
        &evaluation_llm_option,
    );

    let generation_instructions = resolve_inline_or_file(
        generation_instructions,
        generation_instructions_file,
        "generation instructions",
    )?;
    let evaluation_instructions = resolve_inline_or_file(
        evaluation_instructions,
        evaluation_instructions_file,
        "evaluation instructions",
    )?;
    validate_instructions(
        generation_instructions.as_deref(),
        "generation instructions",
    )?;
    validate_instructions(
        evaluation_instructions.as_deref(),
        "evaluation instructions",
    )?;

    let generation_prompt_template = read_optional_text_file(
        generation_prompt_template_file,
        "generation prompt template",
    )?;
    let evaluation_prompt_template = read_optional_text_file(
        evaluation_prompt_template_file,
        "evaluation prompt template",
    )?;
    validate_prompt_template(
        generation_prompt_template.as_deref(),
        "generation prompt template",
    )?;
    validate_prompt_template(
        evaluation_prompt_template.as_deref(),
        "evaluation prompt template",
    )?;
    validate_instruction_template_pair(
        generation_instructions.as_deref(),
        generation_prompt_template.as_deref(),
        "generation",
    )?;
    validate_instruction_template_pair(
        evaluation_instructions.as_deref(),
        evaluation_prompt_template.as_deref(),
        "evaluation",
    )?;

    let mut backend = LlmBackend::new(model)
        .with_timeout(Duration::from_secs(llm_timeout))
        .with_schema_mode(schema_mode)
        .with_model_options(llm_option)
        .with_generation_model_options(generation_llm_option)
        .with_evaluation_model_options(evaluation_llm_option);
    if let Some(value) = llm_executable {
        backend = backend.with_executable(value);
    }
    if let Some(value) = generation_model {
        backend = backend.with_generation_model(value);
    }
    if let Some(value) = evaluation_model {
        backend = backend.with_evaluation_model(value);
    }
    if let Some(value) = generation_reasoning_effort {
        backend = backend.with_generation_reasoning_effort(value);
    }
    if let Some(value) = evaluation_reasoning_effort {
        backend = backend.with_evaluation_reasoning_effort(value);
    }
    if let Some(value) = generation_instructions {
        backend = backend.with_generation_instructions(value);
    }
    if let Some(value) = evaluation_instructions {
        backend = backend.with_evaluation_instructions(value);
    }
    if let Some(value) = generation_prompt_template {
        backend = backend.with_generation_prompt_template(value);
    }
    if let Some(value) = evaluation_prompt_template {
        backend = backend.with_evaluation_prompt_template(value);
    }
    Ok(backend)
}

fn effective_reasoning_effort(
    stage_enabled: bool,
    model: &str,
    explicit: Option<ReasoningEffort>,
    common_options: &[LlmOption],
    stage_options: &[LlmOption],
) -> Option<ReasoningEffort> {
    if explicit.is_some()
        || !stage_enabled
        || model != DEFAULT_MODEL
        || common_options
            .iter()
            .chain(stage_options)
            .any(|option| option.key() == "reasoning_effort")
    {
        explicit
    } else {
        Some(DEFAULT_MODEL_REASONING_EFFORT)
    }
}

fn validate_model_argument(flag: &str, value: &str) -> Fallible<()> {
    if value.trim().is_empty() {
        return fail(format!("{flag} requires an exact model ID"));
    }
    if value.trim() != value {
        return fail(format!("{flag} may not have surrounding whitespace"));
    }
    if value.starts_with('-') || value.chars().any(char::is_control) {
        return fail(format!(
            "{flag} may not look like an option or contain control characters"
        ));
    }
    Ok(())
}

fn resolve_inline_or_file(
    inline: Option<String>,
    file: Option<PathBuf>,
    label: &str,
) -> Fallible<Option<String>> {
    match (inline, file) {
        (Some(_), Some(_)) => fail(format!(
            "inline {label} and a {label} file are mutually exclusive"
        )),
        (Some(value), None) => Ok(Some(value)),
        (None, Some(path)) => read_text_file(&path, label).map(Some),
        (None, None) => Ok(None),
    }
}

fn read_optional_text_file(path: Option<PathBuf>, label: &str) -> Fallible<Option<String>> {
    path.map(|path| read_text_file(&path, label)).transpose()
}

fn read_text_file(path: &Path, label: &str) -> Fallible<String> {
    let initial_metadata = fs::metadata(path).map_err(|error| {
        ErrorReport::new(format!(
            "could not inspect {label} file '{}': {error}",
            path.display()
        ))
    })?;
    if !initial_metadata.is_file() {
        return fail(format!(
            "{label} path '{}' is not a regular file",
            path.display()
        ));
    }
    let file = fs::File::open(path).map_err(|error| {
        ErrorReport::new(format!(
            "could not read {label} file '{}': {error}",
            path.display()
        ))
    })?;
    let metadata = file.metadata().map_err(|error| {
        ErrorReport::new(format!(
            "could not inspect {label} file '{}': {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return fail(format!(
            "{label} path '{}' is not a regular file",
            path.display()
        ));
    }
    if metadata.len() > MAX_CUSTOMIZATION_FILE_BYTES {
        return fail(format!(
            "{label} file '{}' exceeds the {}-byte limit",
            path.display(),
            MAX_CUSTOMIZATION_FILE_BYTES
        ));
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_CUSTOMIZATION_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            ErrorReport::new(format!(
                "could not read {label} file '{}': {error}",
                path.display()
            ))
        })?;
    if bytes.len() as u64 > MAX_CUSTOMIZATION_FILE_BYTES {
        return fail(format!(
            "{label} file '{}' exceeds the {}-byte limit",
            path.display(),
            MAX_CUSTOMIZATION_FILE_BYTES
        ));
    }
    String::from_utf8(bytes).map_err(|error| {
        ErrorReport::new(format!(
            "{label} file '{}' is not valid UTF-8: {error}",
            path.display()
        ))
    })
}

fn validate_instructions(value: Option<&str>, label: &str) -> Fallible<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.trim().is_empty() {
        return fail(format!("{label} may not be blank"));
    }
    if value.contains('\0') {
        return fail(format!("{label} may not contain a NUL byte"));
    }
    Ok(())
}

fn validate_prompt_template(value: Option<&str>, label: &str) -> Fallible<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.trim().is_empty() {
        return fail(format!("{label} may not be blank"));
    }
    if !value.contains(PROMPT_INPUT_JSON_PLACEHOLDER) {
        return fail(format!(
            "{label} must contain {PROMPT_INPUT_JSON_PLACEHOLDER}"
        ));
    }
    Ok(())
}

fn validate_instruction_template_pair(
    instructions: Option<&str>,
    template: Option<&str>,
    stage: &str,
) -> Fallible<()> {
    if instructions.is_some()
        && template.is_some_and(|value| !value.contains(PROMPT_DEFAULT_INSTRUCTIONS_PLACEHOLDER))
    {
        return fail(format!(
            "a {stage} prompt template used with {stage} instructions must contain {PROMPT_DEFAULT_INSTRUCTIONS_PLACEHOLDER}"
        ));
    }
    Ok(())
}

fn parse_specs(path: &Path) -> Fallible<Vec<DrillSpec>> {
    parse_path(path).map_err(|error| ErrorReport::new(format!("parse: {error}")))
}

fn resolve_path(path: Option<PathBuf>) -> Fallible<PathBuf> {
    Ok(match path {
        Some(path) => path,
        None => std::env::current_dir()?,
    })
}

fn collection_root(path: &Path) -> Fallible<&Path> {
    if path.is_file() {
        Ok(path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new(".")))
    } else {
        Ok(path)
    }
}

fn parse_date(value: &str) -> Fallible<Date> {
    match value {
        "today" => Ok(Date::today()),
        "tomorrow" => Ok(Date::tomorrow()),
        value => Date::try_from(value.to_string()),
    }
}

fn parse_eval_repeats(value: &str) -> Result<usize, String> {
    parse_bounded_usize(value, 1, 10, "repeats")
}

fn parse_eval_concurrency(value: &str) -> Result<usize, String> {
    parse_bounded_usize(value, 1, 16, "concurrency")
}

fn parse_bounded_usize(
    value: &str,
    minimum: usize,
    maximum: usize,
    label: &str,
) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{label} must be an integer from {minimum} through {maximum}"))?;
    if !(minimum..=maximum).contains(&parsed) {
        return Err(format!("{label} must be from {minimum} through {maximum}"));
    }
    Ok(parsed)
}

fn print_samples_human(records: &[SampleRecord]) {
    for (index, record) in records.iter().enumerate() {
        if index > 0 {
            println!("\n---\n");
        }
        println!(
            "{} · sample {} · {}",
            terminal_safe(&record.deck),
            record.sample,
            record.spec_hash
        );
        println!("\nQ: {}", terminal_safe(&record.question));
        println!("\nTarget: {}", terminal_safe(&record.target));
        if !record.rubric.is_empty() && record.rubric != record.target {
            println!("\nCriteria: {}", terminal_safe(&record.rubric));
        }
        println!(
            "\nModel: {} (protocol {})",
            terminal_safe(&record.model),
            record.protocol_version
        );
    }
}

fn print_stats_human(stats: StorageStats) {
    println!("Drill specifications\t{}", stats.total_specs);
    println!("Unseen\t{}", stats.unseen_specs);
    println!("Due now\t{}", stats.due_specs);
    println!("Frozen instances\t{}", stats.frozen_instances);
    println!("Attempts\t{}", stats.attempts);
    println!("Reviews\t{}", stats.reviews);
    println!("Completed sessions\t{}", stats.completed_sessions);
}

fn print_evaluation_report(report: &EvaluationReport) {
    println!("ANSWER EVALUATION");
    println!("Calls\t{}", report.calls);
    println!("Usable\t{}", report.usable_calls);
    println!("Errors\t{}", report.errors);
    println!("Binary accuracy\t{:.1}%", report.binary_accuracy.percent);
    println!("Exact accuracy\t{:.1}%", report.exact_accuracy.percent);
    println!("False YEA\t{}", report.false_yea);
    println!("False NAY\t{}", report.false_nay);
    println!(
        "Latency\tp50 {} ms · p95 {} ms · max {} ms",
        report.latency.p50_ms, report.latency.p95_ms, report.latency.max_ms
    );
    println!(
        "Feedback\tmean {:.1} chars · {:.1} words",
        report.feedback.mean_chars, report.feedback.mean_words
    );
    if !report.models.is_empty() {
        println!(
            "Models\t{}",
            report
                .models
                .iter()
                .map(|model| terminal_safe(model))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if !report.domains.is_empty() {
        println!("\nDOMAINS");
        for domain in &report.domains {
            println!(
                "{}\t{:.1}% binary · {:.1}% exact · {}/{} usable",
                terminal_safe(&domain.domain),
                domain.binary_accuracy.percent,
                domain.exact_accuracy.percent,
                domain.usable_calls,
                domain.calls
            );
        }
    }
    let misses = answer_misses(&report.results);
    if !misses.is_empty() {
        println!("\nMISSES");
        for result in misses {
            let outcome = result.outcome.as_ref().expect("miss has an outcome");
            println!(
                "{} · repeat {}\t{}",
                terminal_safe(&result.case_id),
                result.repeat,
                outcome.verdict
            );
        }
    }
    print_eval_errors(
        report
            .results
            .iter()
            .filter_map(|result| result.error.as_ref().map(|error| (&result.case_id, error))),
    );
}

fn print_generation_report(report: &GenerationReport, show_samples: bool) {
    println!("GENERATION EVALUATION");
    println!("Calls\t{}", report.calls);
    println!("Usable\t{}", report.usable_generations);
    println!("Errors\t{}", report.generation_errors);
    println!(
        "Generation success\t{:.1}%",
        report.generation_success_percent
    );
    println!(
        "Generation latency\tp50 {} ms · p95 {} ms · max {} ms",
        report.generation_latency.p50_ms,
        report.generation_latency.p95_ms,
        report.generation_latency.max_ms
    );
    println!(
        "Diversity\t{}/{} unique ({:.1}%)",
        report.lexical_diversity.unique_questions,
        report.lexical_diversity.samples,
        report.lexical_diversity.unique_percent
    );
    if let Some(distance) = report.lexical_diversity.mean_pairwise_distance {
        println!("Pairwise distance\t{distance:.3}");
    }
    if !report.candidate_models.is_empty() {
        println!(
            "Candidate models\t{}",
            report
                .candidate_models
                .iter()
                .map(|model| terminal_safe(model))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if report.judge_requested {
        println!(
            "Judge coverage\t{}/{} ({:.1}%)",
            report.judge_coverage.correct,
            report.judge_coverage.total,
            report.judge_coverage.percent
        );
        println!("Judge errors\t{}", report.judge_errors);
        println!(
            "Quality\t{}/{} judged ({:.1}%) · {:.1}% end-to-end",
            report.quality_passes,
            report.judge_coverage.correct,
            report.quality_percent_of_judged,
            report.end_to_end_quality_percent
        );
        println!(
            "Judge latency\tp50 {} ms · p95 {} ms · max {} ms",
            report.judge_latency.p50_ms, report.judge_latency.p95_ms, report.judge_latency.max_ms
        );
        if !report.judge_models.is_empty() {
            println!(
                "Judge models\t{}",
                report
                    .judge_models
                    .iter()
                    .map(|model| terminal_safe(model))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    let failures = generation_quality_failures(&report.results);
    if !failures.is_empty() {
        println!("\nQUALITY FAILURES");
        for result in failures {
            println!(
                "{} · repeat {}",
                terminal_safe(&result.case_id),
                result.repeat
            );
        }
    }

    print_eval_errors(
        report
            .results
            .iter()
            .filter_map(|result| result.error.as_ref().map(|error| (&result.case_id, error))),
    );
    if show_samples {
        for result in &report.results {
            let Some(sample) = &result.sample else {
                continue;
            };
            println!(
                "\n---\n\n{} · repeat {}\n\nQ: {}\n\nA: {}",
                terminal_safe(&result.case_id),
                result.repeat,
                terminal_safe(&sample.question),
                terminal_safe(&sample.target)
            );
            if sample.rubric != sample.target {
                println!("\nCriteria: {}", terminal_safe(&sample.rubric));
            }
        }
    }
}

fn answer_misses(results: &[EvaluationCallReport]) -> Vec<&EvaluationCallReport> {
    results
        .iter()
        .filter(|result| {
            result
                .outcome
                .as_ref()
                .is_some_and(|outcome| !outcome.binary_correct)
        })
        .collect()
}

fn generation_quality_failures(results: &[GenerationCallReport]) -> Vec<&GenerationCallReport> {
    results
        .iter()
        .filter(|result| result.judge == JudgeOutcome::Fail)
        .collect()
}

fn print_eval_errors<'a>(errors: impl Iterator<Item = (&'a String, &'a String)>) {
    let errors: Vec<_> = errors.collect();
    if errors.is_empty() {
        return;
    }
    println!("\nERRORS");
    for (case_id, error) in errors {
        println!("{}\t{}", terminal_safe(case_id), terminal_safe(error));
    }
}

fn terminal_safe(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn write_executable(path: &Path, contents: &str) {
        use std::os::unix::fs::PermissionsExt;

        fs::write(path, contents).unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions).unwrap();
    }

    fn write_collection(directory: &Path) {
        fs::create_dir_all(directory.join("science")).unwrap();
        fs::create_dir_all(directory.join("arts")).unwrap();
        fs::write(
            directory.join("science/Physics.md"),
            "+++\nname = \"Physics\"\n+++\nQ: Static physics?\nA: Yes.\n",
        )
        .unwrap();
        fs::write(
            directory.join("science/Chemistry.md"),
            "+++\nname = \"Chemistry\"\n+++\nQ: Static chemistry?\nA: Yes.\n",
        )
        .unwrap();
        fs::write(
            directory.join("arts/Poetry.md"),
            "+++\nname = \"Poetry\"\n+++\nQ: Static poetry?\nA: Yes.\n",
        )
        .unwrap();
    }

    #[test]
    fn sample_defaults_are_stable() {
        let cli = Cli::try_parse_from(["hashdrills", "sample", "drills.md"]).unwrap();
        match cli.command {
            Command::Sample {
                path,
                count,
                common,
                generation,
                selection,
                format,
            } => {
                assert_eq!(path, Some(PathBuf::from("drills.md")));
                assert_eq!(count, 3);
                assert_eq!(common.model, DEFAULT_MODEL);
                assert!(common.llm_option.is_empty());
                assert_eq!(common.schema_mode, SchemaMode::Native);
                assert_eq!(common.llm_executable, None);
                assert_eq!(common.llm_timeout, DEFAULT_TIMEOUT.as_secs());
                assert_eq!(generation.generation_model, None);
                assert_eq!(generation.generation_reasoning_effort, None);
                assert_eq!(selection, SelectionArgs::default());
                assert_eq!(format, OutputFormat::Human);
            }
            command => panic!("unexpected command: {command:?}"),
        }
    }

    #[test]
    fn omitted_reasoning_flags_remain_distinct_at_parse_time() {
        let cli = Cli::try_parse_from(["hashdrills", "drill", "drills.md"]).unwrap();
        match cli.command {
            Command::Drill { llm, .. } => {
                assert_eq!(llm.common.model, DEFAULT_MODEL);
                assert_eq!(llm.generation.generation_reasoning_effort, None);
                assert_eq!(llm.evaluation.evaluation_reasoning_effort, None);
            }
            command => panic!("unexpected command: {command:?}"),
        }
    }

    #[test]
    fn benchmarked_default_gets_none_without_leaking_to_other_models() {
        assert_eq!(
            effective_reasoning_effort(true, DEFAULT_MODEL, None, &[], &[]),
            Some(ReasoningEffort::None)
        );
        assert_eq!(
            effective_reasoning_effort(true, "local-model", None, &[], &[]),
            None
        );
        assert_eq!(
            effective_reasoning_effort(false, DEFAULT_MODEL, None, &[], &[]),
            None
        );
        assert_eq!(
            effective_reasoning_effort(true, DEFAULT_MODEL, Some(ReasoningEffort::Low), &[], &[],),
            Some(ReasoningEffort::Low)
        );

        let provider_override = LlmOption::new("reasoning_effort", "high").unwrap();
        assert_eq!(
            effective_reasoning_effort(true, DEFAULT_MODEL, None, &[provider_override], &[],),
            None
        );
    }

    #[test]
    fn omitted_executable_preserves_the_backends_hardened_path_resolution() {
        let cli = Cli::try_parse_from(["hashdrills", "sample", "drills.md"]).unwrap();
        let Command::Sample {
            common, generation, ..
        } = cli.command
        else {
            panic!("expected sample")
        };
        assert_eq!(common.llm_executable, None);
        let backend = build_scoped_backend(common, Some(generation), None).unwrap();
        assert!(
            format!("{backend:?}").contains("resolve_executable_from_path: true"),
            "omitting --llm-executable must retain safe PATH resolution"
        );
    }

    #[test]
    fn drill_accepts_provider_prompt_selection_archive_and_server_controls() {
        let cli = Cli::try_parse_from([
            "hashdrills",
            "drill",
            ".",
            "--model",
            "fallback-model",
            "--generation-model",
            "generator-model",
            "--evaluation-model",
            "judge-model",
            "--generation-reasoning-effort",
            "high",
            "--evaluation-reasoning-effort",
            "low",
            "--llm-option",
            "temperature=0.2",
            "--llm-option",
            "seed=42",
            "--generation-llm-option",
            "temperature=0.8",
            "--evaluation-llm-option",
            "json_object=true",
            "--schema-mode",
            "prompt",
            "--llm-executable",
            "/opt/bin/llm",
            "--llm-timeout",
            "45",
            "--generation-instructions",
            "Vary the surface form.",
            "--evaluation-instructions",
            "Accept equivalent notation.",
            "--generation-prompt-template-file",
            "generate.txt",
            "--evaluation-prompt-template-file",
            "evaluate.txt",
            "--new-card-limit",
            "2",
            "--card-limit",
            "8",
            "--from-deck",
            "Physics",
            "--include-deck",
            "Chemistry",
            "--include-path",
            "science",
            "--exclude-deck",
            "History",
            "--exclude-path",
            "drafts",
            "--save-generated",
            "archive",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--open-browser",
            "false",
            "--public-url",
            "https://drills.example.test",
            "--no-auth",
        ])
        .unwrap();
        match cli.command {
            Command::Drill {
                path,
                llm,
                new_drill_limit,
                drill_limit,
                selection,
                save_generated,
                host,
                port,
                open_browser,
                public_url,
                no_auth,
            } => {
                assert_eq!(path, Some(PathBuf::from(".")));
                assert_eq!(llm.common.model, "fallback-model");
                assert_eq!(
                    llm.generation.generation_model.as_deref(),
                    Some("generator-model")
                );
                assert_eq!(
                    llm.evaluation.evaluation_model.as_deref(),
                    Some("judge-model")
                );
                assert_eq!(
                    llm.generation.generation_reasoning_effort,
                    Some(ReasoningEffort::High)
                );
                assert_eq!(
                    llm.evaluation.evaluation_reasoning_effort,
                    Some(ReasoningEffort::Low)
                );
                assert_eq!(llm.common.llm_option.len(), 2);
                assert_eq!(llm.common.llm_option[0].key(), "temperature");
                assert_eq!(llm.common.llm_option[0].value(), "0.2");
                assert_eq!(llm.generation.generation_llm_option[0].value(), "0.8");
                assert_eq!(llm.evaluation.evaluation_llm_option[0].key(), "json_object");
                assert_eq!(llm.common.schema_mode, SchemaMode::Prompt);
                assert_eq!(
                    llm.common.llm_executable,
                    Some(OsString::from("/opt/bin/llm"))
                );
                assert_eq!(llm.common.llm_timeout, 45);
                assert_eq!(new_drill_limit, Some(2));
                assert_eq!(drill_limit, Some(8));
                assert_eq!(selection.include_deck, ["Physics", "Chemistry"]);
                assert_eq!(selection.include_path, [PathBuf::from("science")]);
                assert_eq!(selection.exclude_deck, ["History"]);
                assert_eq!(selection.exclude_path, [PathBuf::from("drafts")]);
                assert_eq!(save_generated, Some(PathBuf::from("archive")));
                assert_eq!(host, "0.0.0.0");
                assert_eq!(port, 9000);
                assert!(!open_browser);
                assert_eq!(public_url.as_deref(), Some("https://drills.example.test"));
                assert!(no_auth);
            }
            command => panic!("unexpected command: {command:?}"),
        }
    }

    #[test]
    fn eval_answer_defaults_use_only_common_and_evaluation_settings() {
        let cli = Cli::try_parse_from(["hashdrills", "eval", "answers"]).unwrap();
        let Command::Eval {
            suite:
                EvalCommand::Answers {
                    cases,
                    common,
                    evaluation,
                    run,
                },
        } = cli.command
        else {
            panic!("expected answer eval")
        };
        assert_eq!(cases, None);
        assert_eq!(common.model, DEFAULT_MODEL);
        assert_eq!(evaluation.evaluation_model, None);
        assert_eq!(evaluation.evaluation_reasoning_effort, None);
        assert_eq!(run.repeats, 1);
        assert_eq!(run.concurrency, 1);
        assert!(run.case_ids.is_empty());
        assert_eq!(run.seed, 0);
        assert_eq!(run.format, OutputFormat::Human);

        let cli = Cli::try_parse_from(["hashdrills", "eval", "generation"]).unwrap();
        let Command::Eval {
            suite: EvalCommand::Generation { judge, .. },
        } = cli.command
        else {
            panic!("expected generation eval")
        };
        assert_eq!(judge.judge_model, DEFAULT_JUDGE_MODEL);
        assert_eq!(judge.judge_reasoning_effort, ReasoningEffort::High);
        assert_eq!(judge.judge_concurrency, 1);
    }

    #[test]
    fn generation_eval_accepts_candidate_and_fixed_judge_controls() {
        let cli = Cli::try_parse_from([
            "hashdrills",
            "eval",
            "generation",
            "cases.json",
            "--model",
            "candidate-fallback",
            "--generation-model",
            "candidate-model",
            "--generation-reasoning-effort",
            "low",
            "--generation-llm-option",
            "temperature=0.7",
            "--judge-model",
            "judge-model",
            "--judge-reasoning-effort",
            "xhigh",
            "--judge-llm-option",
            "temperature=0",
            "--judge-concurrency",
            "3",
            "--repeats",
            "4",
            "--concurrency",
            "2",
            "--case",
            "physics",
            "--case",
            "poetry",
            "--seed",
            "99",
            "--format",
            "json",
            "--show-samples",
        ])
        .unwrap();
        let Command::Eval {
            suite:
                EvalCommand::Generation {
                    cases,
                    common,
                    generation,
                    run,
                    judge,
                    skip_judge,
                    show_samples,
                },
        } = cli.command
        else {
            panic!("expected generation eval")
        };
        assert_eq!(cases, Some(PathBuf::from("cases.json")));
        assert_eq!(common.model, "candidate-fallback");
        assert_eq!(
            generation.generation_model.as_deref(),
            Some("candidate-model")
        );
        assert_eq!(
            generation.generation_reasoning_effort,
            Some(ReasoningEffort::Low)
        );
        assert_eq!(generation.generation_llm_option[0].value(), "0.7");
        assert_eq!(judge.judge_model, "judge-model");
        assert_eq!(judge.judge_reasoning_effort, ReasoningEffort::Xhigh);
        assert_eq!(judge.judge_llm_option[0].value(), "0");
        assert_eq!(judge.judge_concurrency, 3);
        assert_eq!(run.repeats, 4);
        assert_eq!(run.concurrency, 2);
        assert_eq!(run.case_ids, ["physics", "poetry"]);
        assert_eq!(run.seed, 99);
        assert_eq!(run.format, OutputFormat::Json);
        assert!(!skip_judge);
        assert!(show_samples);
    }

    #[test]
    fn stage_specific_help_surfaces_reject_irrelevant_model_flags() {
        assert!(
            Cli::try_parse_from(["hashdrills", "sample", ".", "--evaluation-model", "unused"])
                .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "hashdrills",
                "eval",
                "answers",
                "--generation-model",
                "unused"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "hashdrills",
                "eval",
                "generation",
                "--evaluation-model",
                "unused"
            ])
            .is_err()
        );
    }

    #[test]
    fn eval_bounds_are_rejected_during_argument_parsing() {
        for (flag, value) in [
            ("--repeats", "0"),
            ("--repeats", "11"),
            ("--concurrency", "0"),
            ("--concurrency", "17"),
            ("--judge-concurrency", "17"),
        ] {
            assert!(
                Cli::try_parse_from(["hashdrills", "eval", "generation", flag, value]).is_err(),
                "{flag}={value} should fail"
            );
        }
    }

    #[test]
    fn selectors_are_available_on_every_collection_command() {
        for command in ["sample", "drill", "due", "stats"] {
            let cli = Cli::try_parse_from([
                "hashdrills",
                command,
                ".",
                "--include-deck",
                "A",
                "--from-deck",
                "B",
                "--include-path",
                "one",
                "--exclude-deck",
                "C",
                "--exclude-path",
                "two",
            ])
            .unwrap();
            let selection = match cli.command {
                Command::Sample { selection, .. }
                | Command::Drill { selection, .. }
                | Command::Due { selection, .. }
                | Command::Stats { selection, .. } => selection,
                command => panic!("unexpected command: {command:?}"),
            };
            assert_eq!(selection.include_deck, ["A", "B"]);
            assert_eq!(selection.include_path, [PathBuf::from("one")]);
            assert_eq!(selection.exclude_deck, ["C"]);
            assert_eq!(selection.exclude_path, [PathBuf::from("two")]);
        }
    }

    #[test]
    fn explicit_none_reasoning_is_distinct_from_an_omitted_flag() {
        let cli = Cli::try_parse_from([
            "hashdrills",
            "sample",
            ".",
            "--generation-reasoning-effort",
            "none",
        ])
        .unwrap();
        let Command::Sample { generation, .. } = cli.command else {
            panic!("expected sample")
        };
        assert_eq!(
            generation.generation_reasoning_effort,
            Some(ReasoningEffort::None)
        );
    }

    #[test]
    fn inline_and_file_instruction_forms_conflict() {
        for (inline, file) in [
            (
                "--generation-instructions",
                "--generation-instructions-file",
            ),
            (
                "--evaluation-instructions",
                "--evaluation-instructions-file",
            ),
        ] {
            let result = Cli::try_parse_from([
                "hashdrills",
                "drill",
                ".",
                inline,
                "inline",
                file,
                "instructions.txt",
            ]);
            assert!(result.is_err(), "{inline} should conflict with {file}");
        }
    }

    #[test]
    fn malformed_provider_options_and_zero_timeout_are_rejected_by_clap() {
        assert!(
            Cli::try_parse_from([
                "hashdrills",
                "sample",
                ".",
                "--llm-option",
                "missing-equals"
            ])
            .is_err()
        );
        assert!(Cli::try_parse_from(["hashdrills", "sample", ".", "--llm-timeout", "0"]).is_err());
    }

    #[test]
    fn typed_reasoning_rejects_an_ambiguous_generic_override() {
        let cli = Cli::try_parse_from([
            "hashdrills",
            "sample",
            ".",
            "--generation-reasoning-effort",
            "low",
            "--llm-option",
            "reasoning_effort=high",
        ])
        .unwrap();
        let Command::Sample {
            common, generation, ..
        } = cli.command
        else {
            panic!("expected sample")
        };
        let error = build_scoped_backend(common, Some(generation), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicts"), "{error}");
    }

    #[test]
    fn model_fallback_and_stage_overrides_reach_the_backend() {
        let cli = Cli::try_parse_from([
            "hashdrills",
            "drill",
            ".",
            "--model",
            "fallback",
            "--generation-model",
            "generator",
            "--evaluation-model",
            "judge",
        ])
        .unwrap();
        let Command::Drill { llm, .. } = cli.command else {
            panic!("expected drill")
        };
        let backend = build_backend(llm).unwrap();
        assert_eq!(backend.model(), "fallback");
        assert_eq!(backend.generation_model(), "generator");
        assert_eq!(backend.evaluation_model(), "judge");
    }

    #[test]
    fn instruction_and_template_files_are_loaded_and_validated() {
        let directory = tempfile::tempdir().unwrap();
        let instructions = directory.path().join("instructions.txt");
        let template = directory.path().join("template.txt");
        fs::write(&instructions, "Prefer boundary cases.").unwrap();
        fs::write(
            &template,
            "CUSTOM\n{{default_instructions}}\n{{input_json}}",
        )
        .unwrap();

        let cli = Cli::try_parse_from([
            "hashdrills",
            "sample",
            ".",
            "--generation-instructions-file",
            instructions.to_str().unwrap(),
            "--generation-prompt-template-file",
            template.to_str().unwrap(),
        ])
        .unwrap();
        let Command::Sample {
            common, generation, ..
        } = cli.command
        else {
            panic!("expected sample")
        };
        build_scoped_backend(common, Some(generation), None).unwrap();

        fs::write(&template, "missing the required placeholder").unwrap();
        let cli = Cli::try_parse_from([
            "hashdrills",
            "sample",
            ".",
            "--generation-prompt-template-file",
            template.to_str().unwrap(),
        ])
        .unwrap();
        let Command::Sample {
            common, generation, ..
        } = cli.command
        else {
            panic!("expected sample")
        };
        let error = build_scoped_backend(common, Some(generation), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must contain {{input_json}}"), "{error}");

        fs::write(&template, "CUSTOM\n{{input_json}}").unwrap();
        let cli = Cli::try_parse_from([
            "hashdrills",
            "sample",
            ".",
            "--generation-instructions",
            "Keep this overlay.",
            "--generation-prompt-template-file",
            template.to_str().unwrap(),
        ])
        .unwrap();
        let Command::Sample {
            common, generation, ..
        } = cli.command
        else {
            panic!("expected sample")
        };
        let error = build_scoped_backend(common, Some(generation), None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("must contain {{default_instructions}}"),
            "{error}"
        );
    }

    #[test]
    fn customization_files_must_be_regular_bounded_utf8_files() {
        let directory = tempfile::tempdir().unwrap();
        let oversized = directory.path().join("oversized.txt");
        fs::write(
            &oversized,
            vec![b'x'; (MAX_CUSTOMIZATION_FILE_BYTES + 1) as usize],
        )
        .unwrap();
        let error = read_text_file(&oversized, "generation instructions")
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds the 65536-byte limit"), "{error}");

        let invalid_utf8 = directory.path().join("invalid.txt");
        fs::write(&invalid_utf8, [0xff, 0xfe]).unwrap();
        let error = read_text_file(&invalid_utf8, "evaluation instructions")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not valid UTF-8"), "{error}");

        let error = read_text_file(directory.path(), "generation instructions")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a regular file"), "{error}");
    }

    #[test]
    fn include_union_is_filtered_then_exclusions_win() {
        let directory = tempfile::tempdir().unwrap();
        write_collection(directory.path());
        let specs = parse_specs(directory.path()).unwrap();
        let selection = SelectionArgs {
            include_deck: vec!["Physics".into(), "Poetry".into()],
            include_path: vec![PathBuf::from("science")],
            exclude_deck: vec!["Chemistry".into()],
            exclude_path: vec![PathBuf::from("arts")],
        };
        let selected = selected_specs(&specs, directory.path(), selection, true).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].deck_name, "Physics");
    }

    #[test]
    fn parses_relative_dates_and_rejects_invalid_dates() {
        assert_eq!(parse_date("today").unwrap(), Date::today());
        assert_eq!(parse_date("tomorrow").unwrap(), Date::tomorrow());
        assert_eq!(parse_date("2026-07-31").unwrap().to_string(), "2026-07-31");
        assert!(parse_date("next week").is_err());
    }

    #[test]
    fn terminal_sanitizer_strips_escape_osc_bell_and_carriage_return() {
        let hostile = "safe\u{1b}]8;;https://evil.invalid\u{7}link\u{1b}]8;;\u{7}\r\nnext";
        let sanitized = terminal_safe(hostile);
        assert_eq!(sanitized, "safe]8;;https://evil.invalidlink]8;;\nnext");
        assert!(
            !sanitized
                .chars()
                .any(|character| { matches!(character, '\u{1b}' | '\u{7}' | '\r') })
        );
    }

    #[test]
    fn human_report_failure_lists_are_private_safe_and_actionable() {
        let answer_results = vec![
            EvaluationCallReport {
                case_id: "hit".into(),
                repeat: 1,
                outcome: Some(crate::evals::EvaluationOutcome {
                    verdict: crate::model::Verdict::Pass,
                    binary_correct: true,
                    exact_correct: true,
                }),
                elapsed_ms: 1,
                error: None,
            },
            EvaluationCallReport {
                case_id: "miss\u{1b}\r".into(),
                repeat: 2,
                outcome: Some(crate::evals::EvaluationOutcome {
                    verdict: crate::model::Verdict::Fail,
                    binary_correct: false,
                    exact_correct: false,
                }),
                elapsed_ms: 2,
                error: None,
            },
        ];
        let misses = answer_misses(&answer_results);
        assert_eq!(misses.len(), 1);
        assert_eq!(terminal_safe(&misses[0].case_id), "miss");
        assert_eq!(misses[0].repeat, 2);

        let generation_results = vec![
            GenerationCallReport {
                case_id: "pass".into(),
                repeat: 1,
                generation: crate::evals::GenerationOutcome::Success,
                judge: JudgeOutcome::Pass,
                generation_ms: 1,
                judge_ms: Some(1),
                error: None,
                sample: None,
            },
            GenerationCallReport {
                case_id: "fail\u{7}".into(),
                repeat: 3,
                generation: crate::evals::GenerationOutcome::Success,
                judge: JudgeOutcome::Fail,
                generation_ms: 1,
                judge_ms: Some(1),
                error: None,
                sample: None,
            },
        ];
        let failures = generation_quality_failures(&generation_results);
        assert_eq!(failures.len(), 1);
        assert_eq!(terminal_safe(&failures[0].case_id), "fail");
        assert_eq!(failures[0].repeat, 3);
    }

    #[tokio::test]
    async fn check_and_static_sample_never_create_a_database() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("static.md");
        std::fs::write(&file, "Q: What is 3 times 4?\nA: 12\n").unwrap();

        run(Cli {
            command: Command::Check {
                path: Some(file.clone()),
            },
        })
        .await
        .unwrap();
        let defaults = Cli::try_parse_from(["hashdrills", "sample", "."]).unwrap();
        let Command::Sample {
            common, generation, ..
        } = defaults.command
        else {
            unreachable!()
        };
        run(Cli {
            command: Command::Sample {
                path: Some(file),
                count: 1,
                common,
                generation,
                selection: SelectionArgs::default(),
                format: OutputFormat::Json,
            },
        })
        .await
        .unwrap();

        assert!(!directory.path().join("hashdrills.db").exists());
    }

    #[tokio::test]
    async fn invalid_selection_precedes_database_and_model_side_effects() {
        for command in ["sample", "drill", "due", "stats"] {
            let directory = tempfile::tempdir().unwrap();
            write_collection(directory.path());
            let mut arguments = vec![
                "hashdrills",
                command,
                directory.path().to_str().unwrap(),
                "--include-deck",
                "Typo",
            ];
            if matches!(command, "sample" | "drill") {
                arguments.extend(["--llm-executable", "/definitely/not/llm"]);
            }
            if command == "drill" {
                arguments.extend(["--open-browser", "false"]);
            }
            let error = run(Cli::try_parse_from(arguments).unwrap())
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains("selection:"), "{command}: {error}");
            assert!(error.contains("Typo"), "{command}: {error}");
            assert!(!directory.path().join("hashdrills.db").exists());
        }
    }

    #[tokio::test]
    async fn invalid_eval_suite_and_case_selection_precede_backend_and_database() {
        let directory = tempfile::tempdir().unwrap();
        let empty_suite = directory.path().join("empty.json");
        fs::write(&empty_suite, "[]").unwrap();
        let cli = Cli::try_parse_from([
            "hashdrills",
            "eval",
            "answers",
            empty_suite.to_str().unwrap(),
            "--llm-executable",
            "/definitely/not/llm",
        ])
        .unwrap();
        let error = run(cli).await.unwrap_err().to_string();
        assert!(error.contains("case array is empty"), "{error}");
        assert!(!directory.path().join("hashdrills.db").exists());

        let cli = Cli::try_parse_from([
            "hashdrills",
            "eval",
            "answers",
            "--case",
            "definitely-not-a-case",
            "--llm-executable",
            "/definitely/not/llm",
        ])
        .unwrap();
        let error = run(cli).await.unwrap_err().to_string();
        assert!(error.contains("unknown case ID"), "{error}");
        assert!(!directory.path().join("hashdrills.db").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn answer_eval_uses_the_production_backend_without_creating_a_database() {
        let directory = tempfile::tempdir().unwrap();
        let cases = directory.path().join("answers.json");
        fs::write(
            &cases,
            r#"[{"id":"one","tags":["test"],"question":"Q?","criteria":"A.","response":"A.","allowed_verdicts":["pass"]}]"#,
        )
        .unwrap();
        let arguments = directory.path().join("arguments.txt");
        let executable = directory.path().join("fake-llm");
        write_executable(
            &executable,
            &format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf '%s' '{{\"verdict\":\"pass\",\"feedback\":\"OK\"}}'\n",
                arguments.display()
            ),
        );

        let cli = Cli::try_parse_from([
            "hashdrills",
            "eval",
            "answers",
            cases.to_str().unwrap(),
            "--evaluation-model",
            "answer-model",
            "--evaluation-reasoning-effort",
            "none",
            "--llm-executable",
            executable.to_str().unwrap(),
            "--format",
            "json",
        ])
        .unwrap();
        run(cli).await.unwrap();

        let arguments = fs::read_to_string(arguments).unwrap();
        assert!(arguments.contains("answer-model"), "{arguments}");
        assert!(arguments.contains("reasoning_effort"), "{arguments}");
        assert!(!directory.path().join("hashdrills.db").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn generation_eval_uses_an_isolated_fixed_judge_backend() {
        let directory = tempfile::tempdir().unwrap();
        let cases = directory.path().join("generation.json");
        fs::write(
            &cases,
            r#"[{"id":"one","domain":"math","goal":null,"question_template":"What is {{a small addition problem}}?","answer_template":"{{the correct sum}}","judge_requirements":"The question and target must form one valid arithmetic problem."}]"#,
        )
        .unwrap();
        let arguments = directory.path().join("arguments.txt");
        let prompts = directory.path().join("prompts.txt");
        let executable = directory.path().join("fake-llm");
        write_executable(
            &executable,
            &format!(
                r#"#!/bin/sh
printf '%s\n' "$@" >> '{}'
printf '%s\n' -- >> '{}'
{{ cat; printf '\n--\n'; }} >> '{}'
model=''
previous=''
for argument in "$@"; do
  if [ "$previous" = '-m' ]; then model="$argument"; break; fi
  previous="$argument"
done
if [ "$model" = '{}' ]; then
  printf '%s' '{{"question_valid":true,"target_correct":true,"aligned":true,"constraints_met":true,"answer_leaked":false,"atomic":true,"feedback":"OK"}}'
else
  printf '%s' '{{"question_replacements":["2 + 2"],"answer_replacements":["4"]}}'
fi
"#,
                arguments.display(),
                arguments.display(),
                prompts.display(),
                DEFAULT_JUDGE_MODEL
            ),
        );

        let cli = Cli::try_parse_from([
            "hashdrills",
            "eval",
            "generation",
            cases.to_str().unwrap(),
            "--generation-model",
            "candidate-model",
            "--generation-instructions",
            "CANDIDATE-ONLY-SENTINEL",
            "--llm-option",
            "common=yes",
            "--generation-llm-option",
            "candidate_only=yes",
            "--llm-executable",
            executable.to_str().unwrap(),
            "--format",
            "json",
        ])
        .unwrap();
        run(cli).await.unwrap();

        let arguments = fs::read_to_string(arguments).unwrap();
        assert!(arguments.contains("candidate-model"), "{arguments}");
        assert!(arguments.contains(DEFAULT_JUDGE_MODEL), "{arguments}");
        assert!(arguments.contains("reasoning_effort\nhigh"), "{arguments}");
        assert_eq!(arguments.matches("common").count(), 2, "{arguments}");
        assert_eq!(
            arguments.matches("candidate_only").count(),
            1,
            "{arguments}"
        );
        let prompts = fs::read_to_string(prompts).unwrap();
        assert_eq!(
            prompts.matches("CANDIDATE-ONLY-SENTINEL").count(),
            1,
            "{prompts}"
        );
        assert!(!directory.path().join("hashdrills.db").exists());
    }

    #[tokio::test]
    async fn storage_commands_reject_a_file_collection_before_creating_a_database() {
        for command in ["drill", "due", "stats"] {
            let directory = tempfile::tempdir().unwrap();
            let file = directory.path().join("One.md");
            fs::write(&file, "Q: Static?\nA: Yes.\n").unwrap();
            let mut arguments = vec!["hashdrills", command, file.to_str().unwrap()];
            if command == "drill" {
                arguments.extend(["--open-browser", "false"]);
            }
            let error = run(Cli::try_parse_from(arguments).unwrap())
                .await
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("requires a collection directory"),
                "{command}: {error}"
            );
            assert!(!directory.path().join("hashdrills.db").exists());
        }
    }

    #[tokio::test]
    async fn nested_archive_destination_is_rejected_before_database_creation() {
        let directory = tempfile::tempdir().unwrap();
        write_collection(directory.path());
        let archive = directory.path().join("private/generated");
        let cli = Cli::try_parse_from([
            "hashdrills",
            "drill",
            directory.path().to_str().unwrap(),
            "--save-generated",
            archive.to_str().unwrap(),
            "--open-browser",
            "false",
        ])
        .unwrap();
        let error = run(cli).await.unwrap_err().to_string();
        assert!(error.contains("must be outside collection root"), "{error}");
        assert!(!directory.path().join("hashdrills.db").exists());
        assert!(!archive.exists());
    }

    #[cfg(unix)]
    #[test]
    fn archive_validation_follows_existing_ancestor_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let collection = directory.path().join("collection");
        let outside = directory.path().join("outside");
        fs::create_dir(&collection).unwrap();
        fs::create_dir(&outside).unwrap();
        symlink(&collection, outside.join("into-collection")).unwrap();
        let error = validate_archive_directory(
            Some(outside.join("into-collection/generated")),
            &collection.canonicalize().unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("must be outside collection root"), "{error}");
    }

    #[tokio::test]
    async fn unreadable_prompt_customization_precedes_database_creation() {
        let directory = tempfile::tempdir().unwrap();
        write_collection(directory.path());
        let missing = directory.path().join("missing.txt");
        let cli = Cli::try_parse_from([
            "hashdrills",
            "drill",
            directory.path().to_str().unwrap(),
            "--generation-instructions-file",
            missing.to_str().unwrap(),
            "--open-browser",
            "false",
        ])
        .unwrap();
        let error = run(cli).await.unwrap_err().to_string();
        assert!(error.contains("could not inspect generation instructions"));
        assert!(!directory.path().join("hashdrills.db").exists());
    }
}
