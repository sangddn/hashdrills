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

//! Assisted, provider-neutral configuration for the `llm` model boundary.
//!
//! Hashdrills stores only non-secret defaults. Provider plugins, aliases,
//! endpoints, and credentials remain owned by Simon Willison's `llm` CLI.

use std::ffi::{OsStr, OsString};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Args, ValueEnum};
use serde::Serialize;

use crate::cli::{DEFAULT_MODEL, DEFAULT_MODEL_REASONING_EFFORT};
use crate::config::UserConfig;
use crate::error::{ErrorReport, Fallible, fail};
use crate::model::{
    DEFAULT_TIMEOUT, LlmBackend, ModelBackend, ReasoningEffort, SchemaMode, Verdict,
    resolve_safe_executable,
};
use crate::spec::{DrillSpec, Template};

const MAX_COMMAND_OUTPUT_BYTES: u64 = 256 * 1024;
const MAX_VERSION_CHARS: usize = 160;
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

/// Arguments accepted by `hashdrills setup`.
#[derive(Args, Clone, Debug, Default)]
pub(crate) struct SetupArgs {
    /// Report configuration and model discovery without changing anything.
    #[arg(long, conflicts_with_all = [
        "reset",
        "model",
        "generation_model",
        "evaluation_model",
        "generation_reasoning_effort",
        "evaluation_reasoning_effort",
        "schema_mode",
        "llm_timeout"
    ])]
    pub(crate) check: bool,

    /// Make one generation and one evaluation request through the configured models.
    #[arg(long)]
    pub(crate) test: bool,

    /// Output format for diagnostics and deterministic setup operations.
    #[arg(long, value_enum, default_value_t = SetupFormat::Human)]
    pub(crate) format: SetupFormat,

    /// Remove Hashdrills' saved non-secret model defaults.
    #[arg(long, conflicts_with_all = [
        "check",
        "test",
        "model",
        "generation_model",
        "evaluation_model",
        "generation_reasoning_effort",
        "evaluation_reasoning_effort",
        "schema_mode",
        "llm_timeout",
        "llm_executable"
    ])]
    pub(crate) reset: bool,

    /// Confirm reset or other explicitly displayed setup changes without prompting.
    #[arg(long, requires = "reset")]
    pub(crate) yes: bool,

    /// Save a fallback model for generation and evaluation.
    #[arg(long, value_parser = parse_model_id)]
    pub(crate) model: Option<String>,

    /// Save a generation-only model override.
    #[arg(long, value_parser = parse_model_id)]
    pub(crate) generation_model: Option<String>,

    /// Save an evaluation-only model override.
    #[arg(long, value_parser = parse_model_id)]
    pub(crate) evaluation_model: Option<String>,

    /// Save the generation reasoning effort.
    #[arg(long)]
    pub(crate) generation_reasoning_effort: Option<ReasoningEffort>,

    /// Save the evaluation reasoning effort.
    #[arg(long)]
    pub(crate) evaluation_reasoning_effort: Option<ReasoningEffort>,

    /// Save how Hashdrills supplies JSON schemas to `llm`.
    #[arg(long)]
    pub(crate) schema_mode: Option<SchemaMode>,

    /// Save the maximum duration of each model request, in seconds.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) llm_timeout: Option<u64>,

    /// Use this exact executable for diagnostics and the optional test only.
    /// It is never persisted.
    #[arg(long, hide = true, value_name = "PATH")]
    pub(crate) llm_executable: Option<OsString>,
}

impl SetupArgs {
    fn has_persistent_overrides(&self) -> bool {
        self.model.is_some()
            || self.generation_model.is_some()
            || self.evaluation_model.is_some()
            || self.generation_reasoning_effort.is_some()
            || self.evaluation_reasoning_effort.is_some()
            || self.schema_mode.is_some()
            || self.llm_timeout.is_some()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub(crate) enum SetupFormat {
    #[default]
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReadinessStatus {
    Verified,
    ConfiguredUntested,
    NotReady,
}

impl ReadinessStatus {
    const fn label(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::ConfiguredUntested => "configured, untested",
            Self::NotReady => "not ready",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct SetupReport {
    status: ReadinessStatus,
    llm: LlmReport,
    generation: StageReport,
    evaluation: StageReport,
    schema_mode: SchemaMode,
    timeout_seconds: u64,
    test: Option<TestReport>,
}

#[derive(Clone, Debug, Serialize)]
struct LlmReport {
    found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    executable: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct StageReport {
    model: String,
    discovered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Clone, Debug, Serialize)]
struct TestReport {
    generation: bool,
    evaluation: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Provider {
    OpenAi,
    OpenRouter,
    Ollama,
    Other,
}

impl Provider {
    const fn plugin(self) -> Option<&'static str> {
        match self {
            Self::OpenRouter => Some("llm-openrouter"),
            Self::Ollama => Some("llm-ollama"),
            Self::OpenAi | Self::Other => None,
        }
    }

    const fn key_name(self) -> Option<&'static str> {
        match self {
            Self::OpenAi => Some("openai"),
            Self::OpenRouter => Some("openrouter"),
            Self::Ollama | Self::Other => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Installer {
    Uv,
    Pipx,
    Brew,
}

impl Installer {
    const fn executable(self) -> &'static str {
        match self {
            Self::Uv => "uv",
            Self::Pipx => "pipx",
            Self::Brew => "brew",
        }
    }

    const fn args(self) -> &'static [&'static str] {
        match self {
            Self::Uv => &["tool", "install", "llm"],
            Self::Pipx => &["install", "llm"],
            Self::Brew => &["install", "llm"],
        }
    }
}

struct Capture {
    status: ExitStatus,
    stdout: Vec<u8>,
}

/// Run setup. No credential value is ever read or persisted by this module.
pub(crate) async fn run(args: SetupArgs) -> Fallible<()> {
    if args.reset {
        return reset(args);
    }

    let interactive = io::stdin().is_terminal() && io::stderr().is_terminal();
    if args.format == SetupFormat::Json
        && !args.check
        && !args.test
        && !args.reset
        && !args.has_persistent_overrides()
    {
        return fail(
            "setup --format json requires --check, --test, --reset, or explicit configuration flags",
        );
    }
    if !interactive && !args.check && !args.test && !args.has_persistent_overrides() {
        return fail(
            "setup needs a terminal; use --check or pass explicit model configuration flags",
        );
    }

    let mut config = UserConfig::load()?;

    if args.check {
        let report = inspect(&config, args.llm_executable.as_deref(), args.test).await?;
        print_report(&report, args.format)?;
        return readiness_result(&report);
    }

    if args.has_persistent_overrides() {
        apply_overrides(&mut config, &args);
        if args.test {
            let report = inspect(&config, args.llm_executable.as_deref(), true).await?;
            print_report(&report, args.format)?;
            readiness_result(&report)?;
        }
        config.save()?;
        if !args.test {
            let report = inspect(&config, args.llm_executable.as_deref(), false).await?;
            print_report(&report, args.format)?;
        }
        return Ok(());
    }

    // `--test` alone is deterministic: test the saved/effective defaults but
    // do not rewrite an otherwise unchanged config file.
    if args.test {
        let report = inspect(&config, args.llm_executable.as_deref(), true).await?;
        print_report(&report, args.format)?;
        return readiness_result(&report);
    }

    interactive_wizard(&mut config, args.llm_executable.as_deref()).await
}

fn reset(args: SetupArgs) -> Fallible<()> {
    let confirmed = if args.yes {
        true
    } else if io::stdin().is_terminal() && io::stderr().is_terminal() {
        confirm("Remove Hashdrills' saved model defaults?", false)?
    } else {
        return fail("setup --reset requires --yes when no terminal is attached");
    };

    if !confirmed {
        return fail("setup reset cancelled");
    }
    UserConfig::reset()?;
    match args.format {
        SetupFormat::Human => println!("Hashdrills configuration reset."),
        SetupFormat::Json => println!("{{\"reset\":true}}"),
    }
    Ok(())
}

fn apply_overrides(config: &mut UserConfig, args: &SetupArgs) {
    if let Some(value) = &args.model {
        config.model = Some(value.clone());
    }
    if let Some(value) = &args.generation_model {
        config.generation_model = Some(value.clone());
    }
    if let Some(value) = &args.evaluation_model {
        config.evaluation_model = Some(value.clone());
    }
    if let Some(value) = args.generation_reasoning_effort {
        config.generation_reasoning_effort = Some(value);
    }
    if let Some(value) = args.evaluation_reasoning_effort {
        config.evaluation_reasoning_effort = Some(value);
    }
    if let Some(value) = args.schema_mode {
        config.schema_mode = Some(value);
    }
    if let Some(value) = args.llm_timeout {
        config.llm_timeout = Some(value);
    }
}

async fn inspect(
    config: &UserConfig,
    explicit_executable: Option<&OsStr>,
    test: bool,
) -> Fallible<SetupReport> {
    let effective = EffectiveConfig::from_user(config);
    let executable = resolve_llm(explicit_executable).ok();
    let version = executable
        .as_deref()
        .and_then(|program| capture(program, &[OsStr::new("--version")]).ok())
        .filter(|output| output.status.success())
        .and_then(|output| sanitize_version(&output.stdout));

    let generation_discovered = executable.as_deref().is_some_and(|program| {
        model_is_discovered(program, &effective.generation_model, effective.schema_mode)
    });
    let evaluation_discovered = executable.as_deref().is_some_and(|program| {
        model_is_discovered(program, &effective.evaluation_model, effective.schema_mode)
    });

    let test_report = if test {
        Some(match executable.as_deref() {
            Some(program) => readiness_test(&effective, program).await,
            None => TestReport {
                generation: false,
                evaluation: false,
                detail: Some("`llm` is not installed or not on a safe PATH".to_string()),
            },
        })
    } else {
        None
    };

    let status = readiness_status(
        executable.is_some(),
        generation_discovered,
        evaluation_discovered,
        test_report.as_ref(),
    );

    Ok(SetupReport {
        status,
        llm: LlmReport {
            found: executable.is_some(),
            executable,
            version,
        },
        generation: StageReport {
            model: effective.generation_model,
            discovered: generation_discovered,
            reasoning_effort: effective.generation_reasoning_effort,
        },
        evaluation: StageReport {
            model: effective.evaluation_model,
            discovered: evaluation_discovered,
            reasoning_effort: effective.evaluation_reasoning_effort,
        },
        schema_mode: effective.schema_mode,
        timeout_seconds: effective.timeout.as_secs(),
        test: test_report,
    })
}

fn readiness_status(
    llm_found: bool,
    generation_discovered: bool,
    evaluation_discovered: bool,
    test_report: Option<&TestReport>,
) -> ReadinessStatus {
    if let Some(test) = test_report {
        if test.generation && test.evaluation {
            ReadinessStatus::Verified
        } else {
            ReadinessStatus::NotReady
        }
    } else if !llm_found || !generation_discovered || !evaluation_discovered {
        ReadinessStatus::NotReady
    } else {
        ReadinessStatus::ConfiguredUntested
    }
}

fn readiness_result(report: &SetupReport) -> Fallible<()> {
    match report.status {
        ReadinessStatus::Verified | ReadinessStatus::ConfiguredUntested => Ok(()),
        ReadinessStatus::NotReady => fail(
            "setup is not ready; install/configure `llm`, select discovered models, or rerun setup",
        ),
    }
}

fn print_report(report: &SetupReport, format: SetupFormat) -> Fallible<()> {
    match format {
        SetupFormat::Json => println!("{}", serde_json::to_string_pretty(report)?),
        SetupFormat::Human => {
            println!("Hashdrills setup");
            if let Some(path) = &report.llm.executable {
                let version = report
                    .llm
                    .version
                    .as_deref()
                    .unwrap_or("version unavailable");
                println!(
                    "llm: {} ({version})",
                    terminal_safe(&path.display().to_string())
                );
            } else {
                println!("llm: not found");
            }
            print_stage("Generation", &report.generation);
            print_stage("Evaluation", &report.evaluation);
            println!("Schema: {}", report.schema_mode);
            println!("Timeout: {}s", report.timeout_seconds);
            if let Some(test) = &report.test {
                println!(
                    "Readiness test: generation {}; evaluation {}",
                    pass_label(test.generation),
                    pass_label(test.evaluation)
                );
                if let Some(detail) = &test.detail {
                    println!("Test detail: {}", terminal_safe(detail));
                }
            }
            println!("Status: {}", report.status.label());
        }
    }
    Ok(())
}

fn print_stage(label: &str, stage: &StageReport) {
    let discovered = if stage.discovered {
        "discovered"
    } else {
        "not discovered"
    };
    match stage.reasoning_effort {
        Some(effort) => println!("{label}: {} ({discovered}; {effort})", stage.model),
        None => println!("{label}: {} ({discovered})", stage.model),
    }
}

const fn pass_label(value: bool) -> &'static str {
    if value { "passed" } else { "failed" }
}

#[derive(Clone, Debug)]
struct EffectiveConfig {
    generation_model: String,
    evaluation_model: String,
    generation_reasoning_effort: Option<ReasoningEffort>,
    evaluation_reasoning_effort: Option<ReasoningEffort>,
    schema_mode: SchemaMode,
    timeout: Duration,
}

impl EffectiveConfig {
    fn from_user(config: &UserConfig) -> Self {
        let shared_model = config.model.as_deref().unwrap_or(DEFAULT_MODEL);
        let generation_model = config
            .generation_model
            .as_deref()
            .unwrap_or(shared_model)
            .to_string();
        let evaluation_model = config
            .evaluation_model
            .as_deref()
            .unwrap_or(shared_model)
            .to_string();
        let generation_reasoning_effort = config.generation_reasoning_effort.or_else(|| {
            (generation_model == DEFAULT_MODEL).then_some(DEFAULT_MODEL_REASONING_EFFORT)
        });
        let evaluation_reasoning_effort = config.evaluation_reasoning_effort.or_else(|| {
            (evaluation_model == DEFAULT_MODEL).then_some(DEFAULT_MODEL_REASONING_EFFORT)
        });
        Self {
            generation_model,
            evaluation_model,
            generation_reasoning_effort,
            evaluation_reasoning_effort,
            schema_mode: config.schema_mode.unwrap_or_default(),
            timeout: Duration::from_secs(config.llm_timeout.unwrap_or(DEFAULT_TIMEOUT.as_secs())),
        }
    }

    fn backend(&self, executable: &Path) -> LlmBackend {
        let mut backend = LlmBackend::new(&self.generation_model)
            .with_generation_model(&self.generation_model)
            .with_evaluation_model(&self.evaluation_model)
            .with_schema_mode(self.schema_mode)
            .with_timeout(self.timeout)
            .with_executable(executable.as_os_str());
        if let Some(effort) = self.generation_reasoning_effort {
            backend = backend.with_generation_reasoning_effort(effort);
        }
        if let Some(effort) = self.evaluation_reasoning_effort {
            backend = backend.with_evaluation_reasoning_effort(effort);
        }
        backend
    }
}

async fn readiness_test(config: &EffectiveConfig, executable: &Path) -> TestReport {
    let question = Template::parse("Reply with {{the exact uppercase word READY}}")
        .expect("the setup generation template is valid");
    let answer = Template::parse("The response must be exactly READY.")
        .expect("the setup evaluation template is valid");
    let spec = DrillSpec::new(
        "Hashdrills setup",
        None,
        PathBuf::from("setup.md"),
        (1, 2),
        Some("Confirm generation and evaluation compatibility".into()),
        question,
        answer,
    );
    let backend = config.backend(executable);
    let generated = match backend.generate(&spec).await {
        Ok(generated) => generated,
        Err(error) => {
            return TestReport {
                generation: false,
                evaluation: false,
                detail: Some(format!("generation: {error}")),
            };
        }
    };
    let evaluation = match backend.evaluate(&generated, "READY").await {
        Ok(evaluation) => evaluation,
        Err(error) => {
            return TestReport {
                generation: true,
                evaluation: false,
                detail: Some(format!("evaluation: {error}")),
            };
        }
    };
    if evaluation.verdict != Verdict::Pass {
        return TestReport {
            generation: true,
            evaluation: false,
            detail: Some(format!(
                "evaluation returned {} instead of pass",
                evaluation.verdict
            )),
        };
    }
    TestReport {
        generation: true,
        evaluation: true,
        detail: None,
    }
}

async fn interactive_wizard(
    config: &mut UserConfig,
    explicit_executable: Option<&OsStr>,
) -> Fallible<()> {
    eprintln!("Hashdrills setup");
    eprintln!("Hashdrills stores model defaults only. `llm` owns plugins and keys.\n");

    let executable = match resolve_llm(explicit_executable) {
        Ok(path) => path,
        Err(_) if explicit_executable.is_some() => {
            return fail("setup: the selected `llm` executable is unavailable");
        }
        Err(_) => install_llm_interactively()?,
    };

    if let Ok(output) = capture(&executable, &[OsStr::new("--version")]) {
        if output.status.success() {
            if let Some(version) = sanitize_version(&output.stdout) {
                eprintln!("Found {version}.\n");
            }
        }
    }

    let provider = choose_provider()?;
    configure_provider(&executable, provider)?;

    show_models(&executable, config.schema_mode.unwrap_or_default());
    configure_models(config)?;
    configure_runtime(config)?;

    let effective = EffectiveConfig::from_user(config);
    let generation_discovered = model_is_discovered(
        &executable,
        &effective.generation_model,
        effective.schema_mode,
    );
    let evaluation_discovered = model_is_discovered(
        &executable,
        &effective.evaluation_model,
        effective.schema_mode,
    );
    if !generation_discovered || !evaluation_discovered {
        eprintln!("\nOne or more exact model IDs were not found in `llm models`.");
        if !confirm("Save this configuration without verification?", false)? {
            return fail("setup cancelled; the previous Hashdrills configuration was preserved");
        }
        config.save()?;
        eprintln!("Configuration saved (untested).");
        return Ok(());
    }

    let verified = if confirm(
        "Run one generation and one evaluation request now? This may incur a small provider charge.",
        true,
    )? {
        let report = readiness_test(&effective, &executable).await;
        if report.generation && report.evaluation {
            eprintln!("Readiness test passed.");
            true
        } else {
            eprintln!(
                "Readiness test failed: {}",
                report.detail.as_deref().unwrap_or("unknown failure")
            );
            false
        }
    } else {
        false
    };

    if !verified
        && !confirm(
            "Save these model defaults without a successful readiness test?",
            false,
        )?
    {
        return fail("setup cancelled; the previous Hashdrills configuration was preserved");
    }

    config.save()?;
    eprintln!(
        "Configuration saved ({}).",
        if verified { "verified" } else { "untested" }
    );
    Ok(())
}

fn install_llm_interactively() -> Fallible<PathBuf> {
    eprintln!("The `llm` CLI was not found on a safe PATH.");
    let installer = [Installer::Uv, Installer::Pipx, Installer::Brew]
        .into_iter()
        .find(|installer| resolve_safe_executable(OsStr::new(installer.executable())).is_ok())
        .ok_or_else(|| {
            ErrorReport::new(
                "setup: install `llm` with uv, pipx, or Homebrew, then rerun `hashdrills setup`",
            )
        })?;
    let manager = resolve_safe_executable(OsStr::new(installer.executable()))?;
    show_command(&manager, installer.args());
    if !confirm("Run this installer command?", false)? {
        return fail("setup cancelled; `llm` was not installed");
    }
    let status = Command::new(&manager)
        .args(installer.args())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return fail(format!(
            "setup: `{}` installer failed with status {}",
            installer.executable(),
            status_label(status)
        ));
    }
    resolve_safe_executable(OsStr::new("llm")).map_err(|_| {
        ErrorReport::new(
            "setup: the installer finished, but `llm` is not visible on this PATH; open a new shell and rerun setup",
        )
    })
}

fn choose_provider() -> Fallible<Provider> {
    eprintln!("Provider:");
    eprintln!("  1  OpenAI (built in)");
    eprintln!("  2  OpenRouter");
    eprintln!("  3  Ollama");
    eprintln!("  4  Another installed `llm` provider");
    loop {
        match prompt("Select [1-4]", Some("1"))?.as_str() {
            "1" => return Ok(Provider::OpenAi),
            "2" => return Ok(Provider::OpenRouter),
            "3" => return Ok(Provider::Ollama),
            "4" => return Ok(Provider::Other),
            _ => eprintln!("Enter 1, 2, 3, or 4."),
        }
    }
}

fn configure_provider(llm: &Path, provider: Provider) -> Fallible<()> {
    if let Some(plugin) = provider.plugin() {
        eprintln!("\nProvider plugins execute third-party Python code in `llm`'s environment.");
        let args = ["install", plugin];
        show_command(llm, &args);
        if confirm("Install or update this provider plugin?", false)? {
            run_inherited(llm, &args, "provider plugin installation")?;
        }
    }

    if let Some(key_name) = provider.key_name() {
        eprintln!(
            "\nHashdrills will not read the key. `llm keys set {key_name}` receives it directly."
        );
        let args = ["keys", "set", key_name];
        show_command(llm, &args);
        if confirm("Set or replace this provider key now?", false)? {
            run_inherited(llm, &args, "provider key setup")?;
        }
    }

    if provider == Provider::OpenRouter {
        let args = ["openrouter", "refresh"];
        show_command(llm, &args);
        if confirm("Refresh OpenRouter's model registry now?", true)? {
            run_inherited(llm, &args, "OpenRouter model refresh")?;
        }
    }

    if provider == Provider::Ollama {
        eprintln!(
            "Hashdrills will not start Ollama or download a model. Ensure the server and your chosen model are already available."
        );
    }
    Ok(())
}

fn show_models(llm: &Path, schema_mode: SchemaMode) {
    let args: &[&str] = match schema_mode {
        SchemaMode::Native => &["models", "--schemas"],
        SchemaMode::Prompt => &["models"],
    };
    match capture_strs(llm, args) {
        Ok(output) if output.status.success() => {
            let listing = terminal_safe(&String::from_utf8_lossy(&output.stdout));
            if !listing.trim().is_empty() {
                eprintln!("\nAvailable models:\n{}", listing.trim_end());
            }
        }
        _ => eprintln!("\nCould not list models; enter an exact `llm` model ID."),
    }
}

fn configure_models(config: &mut UserConfig) -> Fallible<()> {
    let current_shared = config.model.as_deref().unwrap_or(DEFAULT_MODEL);
    let shared = prompt("Shared model", Some(current_shared))?;
    parse_model_id(&shared).map_err(ErrorReport::new)?;
    config.model = Some(shared);

    let generation_default = config.generation_model.as_deref().unwrap_or("-");
    let generation = prompt(
        "Generation-only model (`-` uses shared)",
        Some(generation_default),
    )?;
    config.generation_model = optional_model(&generation)?;

    let evaluation_default = config.evaluation_model.as_deref().unwrap_or("-");
    let evaluation = prompt(
        "Evaluation-only model (`-` uses shared)",
        Some(evaluation_default),
    )?;
    config.evaluation_model = optional_model(&evaluation)?;
    Ok(())
}

fn configure_runtime(config: &mut UserConfig) -> Fallible<()> {
    let generation_default = config
        .generation_reasoning_effort
        .map(|effort| effort.to_string())
        .unwrap_or_else(|| "-".into());
    let generation = prompt(
        "Generation reasoning (`-`, none, low, medium, high, xhigh)",
        Some(&generation_default),
    )?;
    config.generation_reasoning_effort = optional_reasoning(&generation)?;

    let evaluation_default = config
        .evaluation_reasoning_effort
        .map(|effort| effort.to_string())
        .unwrap_or_else(|| "-".into());
    let evaluation = prompt(
        "Evaluation reasoning (`-`, none, low, medium, high, xhigh)",
        Some(&evaluation_default),
    )?;
    config.evaluation_reasoning_effort = optional_reasoning(&evaluation)?;

    let schema_default = config.schema_mode.unwrap_or_default().to_string();
    let schema = prompt("Schema mode (native or prompt)", Some(&schema_default))?;
    config.schema_mode = Some(schema.parse().map_err(ErrorReport::new)?);

    let timeout_default = config
        .llm_timeout
        .unwrap_or(DEFAULT_TIMEOUT.as_secs())
        .to_string();
    let timeout = prompt("Request timeout in seconds", Some(&timeout_default))?;
    let timeout = timeout
        .parse::<u64>()
        .map_err(|_| ErrorReport::new("setup: timeout must be a positive integer"))?;
    if timeout == 0 {
        return fail("setup: timeout must be greater than zero");
    }
    config.llm_timeout = Some(timeout);
    Ok(())
}

fn optional_model(value: &str) -> Fallible<Option<String>> {
    if value == "-" {
        return Ok(None);
    }
    parse_model_id(value).map(Some).map_err(ErrorReport::new)
}

fn optional_reasoning(value: &str) -> Fallible<Option<ReasoningEffort>> {
    if value == "-" {
        return Ok(None);
    }
    value.parse().map(Some).map_err(ErrorReport::new)
}

fn model_is_discovered(llm: &Path, model: &str, schema_mode: SchemaMode) -> bool {
    let schema_arg = OsStr::new("--schemas");
    let model_arg = OsStr::new("-m");
    let args = match schema_mode {
        SchemaMode::Native => vec![
            OsStr::new("models"),
            schema_arg,
            model_arg,
            OsStr::new(model),
        ],
        SchemaMode::Prompt => vec![OsStr::new("models"), model_arg, OsStr::new(model)],
    };
    capture(llm, &args).is_ok_and(|output| {
        output.status.success() && !String::from_utf8_lossy(&output.stdout).trim().is_empty()
    })
}

fn resolve_llm(explicit: Option<&OsStr>) -> Result<PathBuf, ErrorReport> {
    if let Some(path) = explicit {
        if path.is_empty() {
            return Err(ErrorReport::new(
                "setup: the llm executable path may not be empty",
            ));
        }
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err(ErrorReport::new(
                "setup: the llm executable path must be absolute",
            ));
        }
        let canonical = path
            .canonicalize()
            .map_err(|_| ErrorReport::new("setup: the selected llm executable does not exist"))?;
        if !canonical
            .metadata()
            .is_ok_and(|metadata| metadata.is_file())
        {
            return Err(ErrorReport::new(
                "setup: the selected llm executable is not a regular file",
            ));
        }
        return Ok(canonical);
    }
    resolve_safe_executable(OsStr::new("llm")).map_err(Into::into)
}

fn run_inherited(program: &Path, args: &[&str], operation: &str) -> Fallible<()> {
    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        fail(format!(
            "setup: {operation} failed with status {}",
            status_label(status)
        ))
    }
}

fn capture_strs(program: &Path, args: &[&str]) -> io::Result<Capture> {
    let args = args.iter().map(OsStr::new).collect::<Vec<_>>();
    capture(program, &args)
}

fn capture(program: &Path, args: &[&OsStr]) -> io::Result<Capture> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let stdout_pipe = child.stdout.take().expect("piped stdout");
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let mut stdout = Vec::new();
        let result = stdout_pipe
            .take(MAX_COMMAND_OUTPUT_BYTES + 1)
            .read_to_end(&mut stdout)
            .map(|_| stdout);
        let _ = sender.send(result);
    });

    let deadline = Instant::now() + DISCOVERY_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "command exceeded the discovery timeout",
                ));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        }
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    let stdout = receiver
        .recv_timeout(remaining)
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => io::Error::new(
                io::ErrorKind::TimedOut,
                "command output exceeded the discovery timeout",
            ),
            mpsc::RecvTimeoutError::Disconnected => {
                io::Error::other("command output reader stopped unexpectedly")
            }
        })??;
    if stdout.len() as u64 > MAX_COMMAND_OUTPUT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "command output exceeded the safety limit",
        ));
    }
    Ok(Capture { status, stdout })
}

fn prompt(label: &str, default: Option<&str>) -> Fallible<String> {
    match default {
        Some(default) => eprint!("{label} [{default}]: "),
        None => eprint!("{label}: "),
    }
    io::stderr().flush()?;
    let mut value = String::new();
    let count = io::stdin().read_line(&mut value)?;
    if count == 0 {
        return fail("setup cancelled at end of input");
    }
    let value = value.trim().to_string();
    if value.is_empty() {
        default
            .map(ToOwned::to_owned)
            .ok_or_else(|| ErrorReport::new("setup: a value is required"))
    } else {
        Ok(value)
    }
}

fn confirm(label: &str, default: bool) -> Fallible<bool> {
    let hint = if default { "Y/n" } else { "y/N" };
    loop {
        eprint!("{label} [{hint}]: ");
        io::stderr().flush()?;
        let mut answer = String::new();
        let count = io::stdin().read_line(&mut answer)?;
        if count == 0 {
            return fail("setup cancelled at end of input");
        }
        let answer = answer.trim();
        match answer.to_ascii_lowercase().as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => eprintln!("Enter y or n."),
        }
    }
}

fn show_command(program: &Path, args: &[&str]) {
    eprint!("Command: {}", terminal_safe(&program.display().to_string()));
    for arg in args {
        eprint!(" {arg}");
    }
    eprintln!();
}

fn sanitize_version(bytes: &[u8]) -> Option<String> {
    let value = terminal_safe(&String::from_utf8_lossy(bytes));
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let value = value.chars().take(MAX_VERSION_CHARS).collect::<String>();
    (!value.is_empty()).then_some(value)
}

fn terminal_safe(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character == '\n' || character == '\t' || !character.is_control() {
                character
            } else {
                '\u{fffd}'
            }
        })
        .collect()
}

fn status_label(status: ExitStatus) -> String {
    status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".into())
}

fn parse_model_id(value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        return Err("model ID may not be blank".into());
    }
    if value.trim() != value {
        return Err("model ID may not have surrounding whitespace".into());
    }
    if value.starts_with('-') || value.chars().any(char::is_control) {
        return Err("model ID may not look like an option or contain control characters".into());
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        setup: SetupArgs,
    }

    #[test]
    fn parses_deterministic_configuration_surface() {
        let parsed = TestCli::try_parse_from([
            "hashdrills-setup-test",
            "--model",
            "shared",
            "--generation-model",
            "generator",
            "--evaluation-model",
            "evaluator",
            "--generation-reasoning-effort",
            "low",
            "--evaluation-reasoning-effort",
            "high",
            "--schema-mode",
            "prompt",
            "--llm-timeout",
            "9",
            "--test",
            "--format",
            "json",
        ])
        .unwrap()
        .setup;
        assert_eq!(parsed.model.as_deref(), Some("shared"));
        assert_eq!(parsed.generation_model.as_deref(), Some("generator"));
        assert_eq!(parsed.evaluation_model.as_deref(), Some("evaluator"));
        assert_eq!(
            parsed.generation_reasoning_effort,
            Some(ReasoningEffort::Low)
        );
        assert_eq!(
            parsed.evaluation_reasoning_effort,
            Some(ReasoningEffort::High)
        );
        assert_eq!(parsed.schema_mode, Some(SchemaMode::Prompt));
        assert_eq!(parsed.llm_timeout, Some(9));
        assert!(parsed.test);
        assert_eq!(parsed.format, SetupFormat::Json);
    }

    #[test]
    fn rejects_zero_timeout_and_unsafe_model_ids() {
        assert!(TestCli::try_parse_from(["test", "--llm-timeout", "0"]).is_err());
        assert!(TestCli::try_parse_from(["test", "--model", "-unsafe"]).is_err());
        assert!(TestCli::try_parse_from(["test", "--model", " padded "]).is_err());
    }

    #[test]
    fn reset_requires_yes_only_when_supplied_and_rejects_configuration() {
        assert!(TestCli::try_parse_from(["test", "--yes"]).is_err());
        assert!(TestCli::try_parse_from(["test", "--reset", "--model", "other"]).is_err());
        assert!(TestCli::try_parse_from(["test", "--reset", "--yes"]).is_ok());
    }

    #[test]
    fn provider_commands_are_curated_and_never_contain_key_values() {
        assert_eq!(Provider::OpenAi.plugin(), None);
        assert_eq!(Provider::OpenAi.key_name(), Some("openai"));
        assert_eq!(Provider::OpenRouter.plugin(), Some("llm-openrouter"));
        assert_eq!(Provider::OpenRouter.key_name(), Some("openrouter"));
        assert_eq!(Provider::Ollama.plugin(), Some("llm-ollama"));
        assert_eq!(Provider::Ollama.key_name(), None);
        assert_eq!(Provider::Other.plugin(), None);
        assert_eq!(Provider::Other.key_name(), None);
    }

    #[test]
    fn implicit_reasoning_depends_on_each_effective_stage_model() {
        let defaults = EffectiveConfig::from_user(&UserConfig::default());
        assert_eq!(
            defaults.generation_reasoning_effort,
            Some(DEFAULT_MODEL_REASONING_EFFORT)
        );
        assert_eq!(
            defaults.evaluation_reasoning_effort,
            Some(DEFAULT_MODEL_REASONING_EFFORT)
        );

        let custom = EffectiveConfig::from_user(&UserConfig {
            model: Some("local-custom".into()),
            evaluation_model: Some(DEFAULT_MODEL.into()),
            ..UserConfig::default()
        });
        assert_eq!(custom.generation_reasoning_effort, None);
        assert_eq!(
            custom.evaluation_reasoning_effort,
            Some(DEFAULT_MODEL_REASONING_EFFORT)
        );
    }

    #[test]
    fn requested_test_failure_is_never_treated_as_untested_success() {
        let failed = TestReport {
            generation: true,
            evaluation: false,
            detail: Some("evaluation failed".to_string()),
        };
        assert_eq!(
            readiness_status(true, true, true, Some(&failed)),
            ReadinessStatus::NotReady
        );

        let passed = TestReport {
            generation: true,
            evaluation: true,
            detail: None,
        };
        assert_eq!(
            readiness_status(true, true, true, Some(&passed)),
            ReadinessStatus::Verified
        );
        assert_eq!(
            readiness_status(true, true, true, None),
            ReadinessStatus::ConfiguredUntested
        );
    }

    #[cfg(unix)]
    fn write_readiness_llm(directory: &Path, verdict: &str) -> PathBuf {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        let executable = directory.join("fake-llm");
        let script = format!(
            r#"#!/bin/sh
set -eu
test "$1" = "prompt"
test "$2" = "--no-stream"
test "$3" = "--no-log"
test "$4" = "-m"
cat >/dev/null
printf '%s\n' "$5" >> "$(dirname "$0")/calls"
case "$5" in
  setup-generation)
    printf '%s' '{{"question_replacements":["READY"],"answer_replacements":[]}}'
    ;;
  setup-evaluation)
    printf '%s' '{{"verdict":"{verdict}","feedback":"short"}}'
    ;;
  *) exit 42 ;;
esac
"#
        );
        fs::write(&executable, script).unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&executable, permissions).unwrap();
        executable
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn readiness_test_uses_two_no_log_stage_calls_and_requires_pass() {
        let directory = tempfile::tempdir().unwrap();
        let executable = write_readiness_llm(directory.path(), "pass");
        let config = EffectiveConfig {
            generation_model: "setup-generation".into(),
            evaluation_model: "setup-evaluation".into(),
            generation_reasoning_effort: None,
            evaluation_reasoning_effort: None,
            schema_mode: SchemaMode::Native,
            timeout: Duration::from_secs(5),
        };
        let passed = readiness_test(&config, &executable).await;
        assert!(passed.generation);
        assert!(passed.evaluation);
        assert_eq!(passed.detail, None);
        assert_eq!(
            std::fs::read_to_string(directory.path().join("calls")).unwrap(),
            "setup-generation\nsetup-evaluation\n"
        );

        let failed_directory = tempfile::tempdir().unwrap();
        let failed_executable = write_readiness_llm(failed_directory.path(), "partial");
        let failed = readiness_test(&config, &failed_executable).await;
        assert!(failed.generation);
        assert!(!failed.evaluation);
        assert!(
            failed
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("instead of pass"))
        );
    }

    #[test]
    fn model_id_validation_matches_backend_safety_boundary() {
        assert_eq!(
            parse_model_id("openrouter/example/model").unwrap(),
            "openrouter/example/model"
        );
        assert!(parse_model_id("").is_err());
        assert!(parse_model_id(" name").is_err());
        assert!(parse_model_id("--option").is_err());
        assert!(parse_model_id("line\nbreak").is_err());
    }

    #[test]
    fn terminal_output_sanitizer_removes_escape_controls() {
        assert_eq!(terminal_safe("safe\u{1b}[31m\nnext"), "safe�[31m\nnext");
    }
}
