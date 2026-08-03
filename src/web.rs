// Copyright 2025–2026 Fernando Borretti
// Modifications Copyright 2026 Sang Doan
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

//! Local web UI for a Hashdrills practice session.
//!
//! The state machine deliberately keeps a generated instance frozen from the
//! moment it is shown until its attempt has been recorded.  Generation and
//! evaluation are awaited without holding either the session or database
//! mutex.  Every mutating form carries a monotonically increasing instance
//! token, so a stale tab cannot answer or grade a different drill.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use axum::Form;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::Path as AxumPath;
use axum::extract::Query;
use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::http::StatusCode;
use axum::http::header::CACHE_CONTROL;
use axum::http::header::CONTENT_DISPOSITION;
use axum::http::header::CONTENT_LENGTH;
use axum::http::header::CONTENT_SECURITY_POLICY;
use axum::http::header::CONTENT_TYPE;
use axum::http::header::LOCATION;
use axum::http::header::REFERRER_POLICY;
use axum::http::header::SET_COOKIE;
use axum::http::header::X_CONTENT_TYPE_OPTIONS;
use axum::http::header::X_FRAME_OPTIONS;
use axum::middleware::Next;
use axum::response::Html;
use axum::response::IntoResponse;
use axum::response::Redirect;
use axum::response::Response;
use axum::routing::get;
use chrono::Duration;
use maud::DOCTYPE;
use maud::Markup;
use maud::PreEscaped;
use maud::html;
use serde::Deserialize;

use crate::archive::ArchiveTrace;
use crate::archive::ArchivedDrill;
use crate::archive::save_generated_drill;
use crate::auth::AccessControl;
use crate::auth::BOOTSTRAP_PATH;
use crate::auth::LaunchAccess;
use crate::error::Fallible;
use crate::fsrs::Grade;
use crate::media::serve as serve_media;
use crate::model::Evaluation;
use crate::model::GeneratedInstance;
use crate::model::ModelBackend;
use crate::model::Verdict;
use crate::render::markdown_to_html;
use crate::render::validate_media_request;
use crate::spec::DrillSpec;
use crate::spec::SpecHash;
use crate::static_assets;
use crate::storage::CompletionHistory;
use crate::storage::ReviewResolution;
use crate::storage::Storage;
use crate::transcript::SESSION_LOG_FORMAT_VERSION;
use crate::transcript::SESSION_LOG_KIND;
use crate::transcript::SESSION_LOG_KIND_MARKER_KEY;
use crate::transcript::SESSION_LOG_VERSION_MARKER_KEY;
use crate::types::date::Date;
use crate::types::timestamp::Timestamp;

const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_MACROS_BYTES: u64 = 16 * 1024;
const MAX_MACROS: usize = 128;
const MAX_MACRO_NAME_BYTES: usize = 65;
const MAX_MACRO_DEFINITION_BYTES: usize = 4 * 1024;
/// Maximum size of one generated in-memory session transcript.
pub const MAX_SESSION_TRANSCRIPT_BYTES: usize = 16 * 1024 * 1024;

/// Configuration for one local practice server.
pub struct ServerConfig {
    /// Specs parsed from the current collection. Historical database rows are
    /// intersected with this set before the queue is built.
    pub specs: Vec<DrillSpec>,
    pub storage: Storage,
    /// Canonical root used for deck-relative rich media and the allowlisted
    /// media route.
    pub collection_root: PathBuf,
    pub backend: Arc<dyn ModelBackend>,
    pub host: String,
    pub port: u16,
    /// Public browser origin used for phone/Tailscale access. A wildcard bind
    /// requires this because wildcard addresses are not valid URLs or Hosts.
    pub public_url: Option<String>,
    /// Open the authenticated local URL in the default browser after binding.
    pub open_browser: bool,
    /// Launch-scoped bearer authentication. This should be true unless the
    /// caller has explicitly requested `--no-auth` on a loopback socket.
    pub auth_enabled: bool,
    pub deck_filter: Option<String>,
    /// Optional plaintext destination for accepted generated instances.
    pub archive_dir: Option<PathBuf>,
    /// Maximum total number of specs in this session.
    pub due_limit: Option<usize>,
    /// Maximum number of unseen specs in this session.
    pub new_limit: Option<usize>,
}

/// Narrow synchronous persistence boundary used by the async UI.
///
/// The concrete implementation locks SQLite only for the duration of one
/// synchronous call. Tests use a deterministic in-memory recorder.
trait SessionRecorder: Send + Sync {
    fn freeze_instance(
        &self,
        session_id: i64,
        spec: &DrillSpec,
        instance: &GeneratedInstance,
        at: Timestamp,
    ) -> Fallible<i64>;

    fn record_attempt(
        &self,
        instance_id: i64,
        response: &str,
        learner_grade: Grade,
        at: Timestamp,
    ) -> Fallible<i64>;

    fn attach_evaluation(
        &self,
        attempt_id: i64,
        evaluation: &Evaluation,
        at: Timestamp,
    ) -> Fallible<()>;

    fn accept_grade(
        &self,
        attempt_id: i64,
        grade: Grade,
        resolution: ReviewResolution,
        at: Timestamp,
    ) -> Fallible<i64>;

    fn undo_review(&self, review_id: i64, at: Timestamp) -> Fallible<()>;

    fn finish_session(&self, session_id: i64, at: Timestamp) -> Fallible<()>;

    fn completion_history_for_specs(
        &self,
        today: Date,
        specs: &[DrillSpec],
    ) -> Fallible<CompletionHistory>;
}

impl SessionRecorder for Mutex<Storage> {
    fn freeze_instance(
        &self,
        session_id: i64,
        spec: &DrillSpec,
        instance: &GeneratedInstance,
        at: Timestamp,
    ) -> Fallible<i64> {
        self.lock()
            .expect("storage mutex poisoned")
            .freeze_instance(session_id, spec.hash(), instance, at)
    }

    fn record_attempt(
        &self,
        instance_id: i64,
        response: &str,
        learner_grade: Grade,
        at: Timestamp,
    ) -> Fallible<i64> {
        Ok(self
            .lock()
            .expect("storage mutex poisoned")
            .record_attempt(instance_id, response, learner_grade, at)?
            .attempt_id)
    }

    fn attach_evaluation(
        &self,
        attempt_id: i64,
        evaluation: &Evaluation,
        at: Timestamp,
    ) -> Fallible<()> {
        self.lock()
            .expect("storage mutex poisoned")
            .attach_evaluation(attempt_id, evaluation, at)?;
        Ok(())
    }

    fn accept_grade(
        &self,
        attempt_id: i64,
        grade: Grade,
        resolution: ReviewResolution,
        at: Timestamp,
    ) -> Fallible<i64> {
        let review = self
            .lock()
            .expect("storage mutex poisoned")
            .accept_grade(attempt_id, grade, resolution, at)?;
        Ok(review.review_id)
    }

    fn undo_review(&self, review_id: i64, at: Timestamp) -> Fallible<()> {
        self.lock()
            .expect("storage mutex poisoned")
            .undo_review(review_id, at)?;
        Ok(())
    }

    fn finish_session(&self, session_id: i64, at: Timestamp) -> Fallible<()> {
        self.lock()
            .expect("storage mutex poisoned")
            .finish_session(session_id, at)
    }

    fn completion_history_for_specs(
        &self,
        today: Date,
        specs: &[DrillSpec],
    ) -> Fallible<CompletionHistory> {
        self.lock()
            .expect("storage mutex poisoned")
            .completion_history_for_specs(today, specs)
    }
}

#[derive(Clone)]
struct AppState {
    access: AccessControl,
    backend: Arc<dyn ModelBackend>,
    recorder: Arc<dyn SessionRecorder>,
    specs: Arc<Vec<DrillSpec>>,
    collection_root: Arc<PathBuf>,
    archive_dir: Option<Arc<PathBuf>>,
    macros: Arc<HashMap<String, String>>,
    shutdown_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    mutable: Arc<Mutex<SessionState>>,
}

struct SessionState {
    session_id: i64,
    started_at: Timestamp,
    finished_at: Option<Timestamp>,
    queue: VecDeque<DrillSpec>,
    total: usize,
    completed: usize,
    next_token: u64,
    current: Option<Current>,
    suspended_current: Option<Current>,
    completed_reviews: Vec<CompletedReview>,
    transcript_entries: Vec<SessionTranscriptEntry>,
    undo_in_progress: bool,
    finish_recorded: bool,
    finish_in_progress: bool,
    finish_error: Option<String>,
    completion_history: Option<CompletionHistory>,
    history_error: Option<String>,
    archive_publication: ArchivePublication,
    archive_error: Option<String>,
    shutdown_requested: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArchivePublication {
    NotStarted,
    Publishing,
    Failed,
    Published,
}

impl ArchivePublication {
    fn locks_undo(self) -> bool {
        self != Self::NotStarted
    }
}

#[derive(Clone)]
struct FrozenQuestion {
    token: u64,
    spec: DrillSpec,
    instance_id: i64,
    generated_at: Timestamp,
    instance: GeneratedInstance,
}

#[derive(Clone)]
struct SubmittedAnswer {
    frozen: FrozenQuestion,
    response: String,
    notice: Option<String>,
}

#[derive(Clone)]
struct RatedAnswer {
    submitted: SubmittedAnswer,
    learner_grade: Grade,
    attempt_id: i64,
}

#[derive(Clone)]
enum CheckResult {
    Skipped,
    Evaluation(Evaluation),
    Error(String),
}

#[derive(Clone)]
struct ReviewResult {
    rated: RatedAnswer,
    check: CheckResult,
    default_grade: Option<Grade>,
    notice: Option<String>,
    regrading: bool,
}

#[derive(Clone)]
struct CompletedReview {
    result: ReviewResult,
    review_id: i64,
    reviewed_at: Timestamp,
    accepted_grade: Grade,
    resolution: ReviewResolution,
}

#[derive(Clone)]
enum SessionTranscriptEntry {
    Reviewed(CompletedReview),
    Skipped(SkippedDrill),
}

#[derive(Clone)]
enum SkippedDrill {
    Ready {
        frozen: FrozenQuestion,
        skipped_at: Timestamp,
    },
    Unresolved {
        result: ReviewResult,
        skipped_at: Timestamp,
    },
    GenerationFailed {
        spec: DrillSpec,
        skipped_at: Timestamp,
    },
}

enum Current {
    Generating {
        token: u64,
        spec: DrillSpec,
    },
    Ready {
        frozen: FrozenQuestion,
        draft: String,
        notice: Option<String>,
    },
    Rating(SubmittedAnswer),
    Recording {
        submitted: SubmittedAnswer,
        grade: Grade,
    },
    Evaluating {
        rated: RatedAnswer,
    },
    RecordingEvaluation {
        rated: RatedAnswer,
        evaluation: Evaluation,
    },
    Result(ReviewResult),
    Saving {
        result: ReviewResult,
        grade: Grade,
    },
    GenerationFailed {
        token: u64,
        spec: DrillSpec,
        message: String,
    },
}

#[derive(Deserialize)]
struct TokenForm {
    #[serde(default)]
    csrf_token: String,
    session_id: i64,
    token: u64,
    instance_id: Option<i64>,
}

#[derive(Deserialize)]
struct AnswerForm {
    #[serde(default)]
    csrf_token: String,
    session_id: i64,
    token: u64,
    instance_id: i64,
    response: String,
}

#[derive(Deserialize)]
struct RateForm {
    #[serde(default)]
    csrf_token: String,
    session_id: i64,
    token: u64,
    instance_id: i64,
    grade: String,
}

#[derive(Deserialize)]
struct AcceptForm {
    #[serde(default)]
    csrf_token: String,
    session_id: i64,
    token: u64,
    instance_id: i64,
    action: String,
    grade: Option<String>,
}

#[derive(Deserialize)]
struct UndoForm {
    #[serde(default)]
    csrf_token: String,
    session_id: i64,
    review_id: i64,
    current_token: Option<u64>,
    current_instance_id: Option<i64>,
    current_draft: Option<String>,
}

#[derive(Deserialize)]
struct SessionForm {
    #[serde(default)]
    csrf_token: String,
    session_id: i64,
}

#[derive(Deserialize)]
struct BootstrapQuery {
    access_token: String,
}

/// Start the local practice UI and serve until interrupted.
pub async fn start_server(mut config: ServerConfig) -> Fallible<()> {
    // Bind and validate the access envelope before opening a session or
    // changing collection registration state.
    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port)).await?;
    let bound_address = listener.local_addr()?;
    let launch_access = LaunchAccess::new(
        config.auth_enabled,
        &config.host,
        bound_address,
        config.public_url.as_deref(),
    )?;

    config
        .storage
        .register_specs(&config.specs, Timestamp::now())?;
    let queue = build_queue(
        &config.storage,
        &config.specs,
        config.deck_filter.as_deref(),
        config.due_limit,
        config.new_limit,
    )?;
    let started_at = Timestamp::now();
    let session_id = config.storage.begin_session(started_at)?;
    let recorder: Arc<dyn SessionRecorder> = Arc::new(Mutex::new(config.storage));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let state = AppState::new(
        launch_access.control.clone(),
        session_id,
        queue,
        config.specs,
        config.collection_root,
        config.archive_dir,
        config.backend,
        recorder,
        started_at,
        Some(shutdown_tx),
    );
    let queue_is_empty = {
        state
            .mutable
            .lock()
            .expect("session mutex poisoned")
            .queue
            .is_empty()
    };
    if queue_is_empty {
        finish_session(&state)?;
    }
    let shutdown_state = state.clone();
    let app = router(state);
    log::info!(
        "Hashdrills listening on http://{}:{}/",
        config.host,
        bound_address.port()
    );
    println!("Hashdrills: {}", launch_access.browser_url);
    if let Some(public_url) = &launch_access.public_url {
        println!("Phone:      {public_url}");
    }
    if config.auth_enabled {
        println!(
            "Anyone with this link can control this launch. It expires when Hashdrills exits."
        );
    }
    if config.open_browser {
        let readiness_host = launch_access.readiness_host;
        let browser_url = launch_access.browser_url;
        drop(tokio::spawn(open_browser_when_ready(
            readiness_host,
            bound_address.port(),
            browser_url,
        )));
    }
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = shutdown_rx => {}
            }
        })
        .await;
    let finish_result = finish_session_on_shutdown(&shutdown_state);
    result?;
    finish_result?;
    Ok(())
}

async fn open_browser_when_ready(host: String, port: u16, url: String) {
    for _ in 0..100 {
        if tokio::net::TcpStream::connect((host.as_str(), port))
            .await
            .is_ok()
        {
            let _ = tokio::task::spawn_blocking(move || open::that(url)).await;
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    eprintln!("hashdrills: web server did not become ready in time to open the browser");
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        access: AccessControl,
        session_id: i64,
        queue: Vec<DrillSpec>,
        specs: Vec<DrillSpec>,
        collection_root: PathBuf,
        archive_dir: Option<PathBuf>,
        backend: Arc<dyn ModelBackend>,
        recorder: Arc<dyn SessionRecorder>,
        started_at: Timestamp,
        shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Self {
        let total = queue.len();
        let macros = load_macros(&collection_root);
        Self {
            access,
            backend,
            recorder,
            specs: Arc::new(specs),
            collection_root: Arc::new(collection_root),
            archive_dir: archive_dir.map(Arc::new),
            macros: Arc::new(macros),
            shutdown_tx: Arc::new(Mutex::new(shutdown_tx)),
            mutable: Arc::new(Mutex::new(SessionState {
                session_id,
                started_at,
                finished_at: None,
                queue: queue.into(),
                total,
                completed: 0,
                next_token: 1,
                current: None,
                suspended_current: None,
                completed_reviews: Vec::new(),
                transcript_entries: Vec::new(),
                undo_in_progress: false,
                finish_recorded: false,
                finish_in_progress: false,
                finish_error: None,
                completion_history: None,
                history_error: None,
                archive_publication: ArchivePublication::NotStarted,
                archive_error: None,
                shutdown_requested: false,
            })),
        }
    }
}

fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/file/{*path}", get(media_file))
        .route(static_assets::KATEX_CSS_URL, get(katex_css))
        .route(
            static_assets::KATEX_JS_URL,
            get(static_assets::katex_js_handler),
        )
        .route(
            static_assets::KATEX_MHCHEM_JS_URL,
            get(static_assets::katex_mhchem_js_handler),
        )
        .route(
            static_assets::KATEX_FONT_ROUTE,
            get(static_assets::katex_font_handler),
        )
        .route(
            static_assets::HIGHLIGHT_CSS_URL,
            get(static_assets::highlight_css_handler),
        )
        .route(
            static_assets::HIGHLIGHT_JS_URL,
            get(static_assets::highlight_js_handler),
        )
        .route("/assets/macros.json", get(macros_json))
        .route("/answer", axum::routing::post(answer))
        .route("/rate", axum::routing::post(rate))
        .route("/accept", axum::routing::post(accept))
        .route("/back", axum::routing::post(back))
        .route("/retry-evaluation", axum::routing::post(retry_evaluation))
        .route("/undo", axum::routing::post(undo))
        .route("/session.md", get(download_session_transcript))
        .route("/shutdown", axum::routing::post(shutdown))
        .route("/retry-finish", axum::routing::post(retry_finish))
        .route("/regenerate", axum::routing::post(regenerate))
        .route("/skip", axum::routing::post(skip))
        .layer(DefaultBodyLimit::max(MAX_RESPONSE_BYTES));
    let app = if state.access.base_path().is_empty() {
        Router::new()
            .route(BOOTSTRAP_PATH, get(bootstrap_access))
            .route("/", get(root))
            .merge(protected)
    } else {
        let root_path = state.access.root_path();
        Router::new()
            .route(BOOTSTRAP_PATH, get(bootstrap_access))
            .route(&root_path, get(root))
            .nest(state.access.base_path(), protected)
    };
    app.layer(axum::middleware::from_fn_with_state(
        state.clone(),
        security_gate,
    ))
    .with_state(state)
}

async fn security_gate(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let is_bootstrap = request.method() == Method::GET && request.uri().path() == BOOTSTRAP_PATH;
    let response = if state.access.bypasses_security() {
        next.run(request).await
    } else {
        if !state.access.host_is_allowed(request.headers()) {
            security_error(StatusCode::MISDIRECTED_REQUEST, "Misdirected Request")
        } else if state.access.enabled() && is_bootstrap {
            next.run(request).await
        } else if !state.access.cookie_is_valid(request.headers()) {
            security_error(
                StatusCode::UNAUTHORIZED,
                "Unauthorized. Open the launch URL printed by Hashdrills.",
            )
        } else if request.method() == Method::POST
            && !state.access.post_origin_is_allowed(request.headers())
        {
            security_error(StatusCode::FORBIDDEN, "Forbidden")
        } else {
            next.run(request).await
        }
    };
    with_security_headers(response, &state.access)
}

async fn bootstrap_access(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<BootstrapQuery>,
) -> Response {
    let Some(cookie) = state.access.bootstrap_cookie(&query.access_token, &headers) else {
        return security_error(StatusCode::UNAUTHORIZED, "Unauthorized");
    };
    let mut response = Redirect::to(&state.access.root_path()).into_response();
    response.headers_mut().insert(SET_COOKIE, cookie);
    response
}

fn security_error(status: StatusCode, message: &'static str) -> Response {
    (
        status,
        [(CONTENT_TYPE, "text/plain; charset=utf-8")],
        message,
    )
        .into_response()
}

fn with_security_headers(mut response: Response, access: &AccessControl) -> Response {
    let headers = response.headers_mut();
    if let Some(location) = headers
        .get(LOCATION)
        .and_then(|value| value.to_str().ok())
        .filter(|location| location.starts_with('/') && *location != BOOTSTRAP_PATH)
    {
        if !access.base_path().is_empty() && !location.starts_with(access.base_path()) {
            let location = format!("{}{}", access.base_path(), location);
            if let Ok(location) = HeaderValue::from_str(&location) {
                headers.insert(LOCATION, location);
            }
        }
    }
    if !headers.contains_key(CACHE_CONTROL) {
        headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    headers.insert(REFERRER_POLICY, HeaderValue::from_static("strict-origin"));
    headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    headers.insert(X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        axum::http::header::HeaderName::from_static("cross-origin-resource-policy"),
        HeaderValue::from_static("same-origin"),
    );
    if !headers.contains_key(CONTENT_SECURITY_POLICY) {
        headers.insert(
            CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'",
            ),
        );
    }
    response
}

async fn download_session_transcript(State(state): State<AppState>) -> Response {
    let (document, finished_at) = {
        let session = state.mutable.lock().expect("session mutex poisoned");
        if !session_transcript_is_ready(&session) {
            return security_error(
                StatusCode::CONFLICT,
                "Session transcript is available after session completion.",
            );
        }
        if session.transcript_entries.len() != session.completed {
            log::error!(
                "Session {} transcript has {} entries for {} resolved drills",
                session.session_id,
                session.transcript_entries.len(),
                session.completed
            );
            return security_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "The completed session transcript is inconsistent.",
            );
        }
        let finished_at = session
            .finished_at
            .expect("transcript readiness requires a finish time");
        let document = match render_session_transcript(&session, finished_at) {
            Ok(document) => document,
            Err(TranscriptTooLarge) => {
                return security_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Session transcript exceeds the 16 MiB download limit.",
                );
            }
        };
        (document, finished_at)
    };

    let filename = transcript_filename(finished_at);
    let disposition = format!("attachment; filename=\"{filename}\"");
    let content_length = document.len();
    let mut response = document.into_response();
    let headers = response.headers_mut();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/markdown; charset=utf-8"),
    );
    headers.insert(
        CONTENT_DISPOSITION,
        HeaderValue::from_str(&disposition).expect("generated filename is header-safe"),
    );
    headers.insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&content_length.to_string())
            .expect("decimal content length is header-safe"),
    );
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("private, no-store"));
    response
}

fn session_transcript_is_ready(session: &SessionState) -> bool {
    session.finish_recorded
        && session.finished_at.is_some()
        && session.queue.is_empty()
        && session.current.is_none()
        && session.suspended_current.is_none()
        && session.completed == session.total
        && !session.undo_in_progress
        && !session.finish_in_progress
        && session.finish_error.is_none()
        && !session.shutdown_requested
}

fn transcript_filename(finished_at: Timestamp) -> String {
    let timestamp = finished_at.to_string();
    let safe = timestamp
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    format!("hashdrills-session-{safe}.md")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TranscriptTooLarge;

struct TranscriptWriter {
    document: String,
}

impl TranscriptWriter {
    fn new() -> Self {
        Self {
            document: String::new(),
        }
    }

    fn push(&mut self, value: &str) -> Result<(), TranscriptTooLarge> {
        let length = self
            .document
            .len()
            .checked_add(value.len())
            .ok_or(TranscriptTooLarge)?;
        if length > MAX_SESSION_TRANSCRIPT_BYTES {
            return Err(TranscriptTooLarge);
        }
        self.document.push_str(value);
        Ok(())
    }

    fn line(&mut self, value: &str) -> Result<(), TranscriptTooLarge> {
        self.push(value)?;
        self.push("\n")
    }

    fn fenced(&mut self, label: &str, value: &str) -> Result<(), TranscriptTooLarge> {
        self.push("\n### ")?;
        self.line(label)?;
        self.push("\n")?;
        let fence_length = longest_backtick_run(value).saturating_add(1).max(3);
        let escaped_length = transcript_text_length(value)?;
        let remaining = MAX_SESSION_TRANSCRIPT_BYTES.saturating_sub(self.document.len());
        let required = fence_length
            .checked_mul(2)
            .and_then(|length| length.checked_add(escaped_length))
            .and_then(|length| length.checked_add(8))
            .ok_or(TranscriptTooLarge)?;
        if required > remaining {
            return Err(TranscriptTooLarge);
        }
        let fence = "`".repeat(fence_length);
        self.push(&fence)?;
        self.push("text\n")?;
        self.push_transcript_text(value)?;
        if !value.ends_with('\n') {
            self.push("\n")?;
        }
        self.push(&fence)?;
        self.push("\n")
    }

    fn finish(self) -> String {
        self.document
    }

    fn push_transcript_text(&mut self, value: &str) -> Result<(), TranscriptTooLarge> {
        let mut encoded = [0_u8; 4];
        for character in value.chars() {
            if transcript_control_needs_escape(character) {
                self.push(&format!("\\u{{{:04X}}}", character as u32))?;
            } else {
                self.push(character.encode_utf8(&mut encoded))?;
            }
        }
        Ok(())
    }
}

fn longest_backtick_run(value: &str) -> usize {
    let mut longest = 0;
    let mut current = 0;
    for byte in value.bytes() {
        if byte == b'`' {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest
}

fn transcript_control_needs_escape(character: char) -> bool {
    matches!(character, '\u{0000}'..='\u{001F}' | '\u{007F}'..='\u{009F}')
        && !matches!(character, '\n' | '\t')
}

fn transcript_text_length(value: &str) -> Result<usize, TranscriptTooLarge> {
    value.chars().try_fold(0_usize, |length, character| {
        length
            .checked_add(if transcript_control_needs_escape(character) {
                8
            } else {
                character.len_utf8()
            })
            .ok_or(TranscriptTooLarge)
    })
}

fn render_session_transcript(
    session: &SessionState,
    finished_at: Timestamp,
) -> Result<String, TranscriptTooLarge> {
    let mut output = TranscriptWriter::new();
    output.line("+++")?;
    output.line(&format!(
        "{SESSION_LOG_KIND_MARKER_KEY} = {SESSION_LOG_KIND:?}"
    ))?;
    output.line(&format!(
        "{SESSION_LOG_VERSION_MARKER_KEY} = {SESSION_LOG_FORMAT_VERSION}"
    ))?;
    output.line("+++")?;
    output.line("")?;
    output.line("# Hashdrills session transcript")?;
    output.line("")?;
    output.line("> Private learning data. Dynamic content is fenced as untrusted text.")?;
    output.line("")?;
    output.line(&format!("- Started: {}", session.started_at))?;
    output.line(&format!("- Finished: {finished_at}"))?;
    output.line(&format!(
        "- Duration: {}",
        format_duration(session.started_at, finished_at)
    ))?;
    output.line(&format!("- Resolved: {}", session.transcript_entries.len()))?;
    output.line(&format!("- Reviewed: {}", session.completed_reviews.len()))?;
    output.line(&format!(
        "- Skipped: {}",
        session
            .transcript_entries
            .len()
            .saturating_sub(session.completed_reviews.len())
    ))?;

    for (index, entry) in session.transcript_entries.iter().enumerate() {
        output.line("")?;
        output.line(&format!("## Entry {}", index + 1))?;
        match entry {
            SessionTranscriptEntry::Reviewed(review) => {
                render_reviewed_transcript_entry(&mut output, review)?;
            }
            SessionTranscriptEntry::Skipped(skipped) => {
                render_skipped_transcript_entry(&mut output, skipped)?;
            }
        }
    }
    Ok(output.finish())
}

fn render_reviewed_transcript_entry(
    output: &mut TranscriptWriter,
    review: &CompletedReview,
) -> Result<(), TranscriptTooLarge> {
    let rated = &review.result.rated;
    let frozen = &rated.submitted.frozen;
    output.line("")?;
    output.line("- Status: reviewed")?;
    output.line(&format!("- Self-rating: {}", rated.learner_grade.as_str()))?;
    output.line(&format!(
        "- Effective rating: {}",
        review.accepted_grade.as_str()
    ))?;
    output.line(&format!(
        "- Resolution: {}",
        review_resolution_label(review.resolution)
    ))?;
    output.line(&format!(
        "- Override: {}",
        if review.resolution == ReviewResolution::UserOverride {
            "yes"
        } else {
            "no"
        }
    ))?;
    output.line(&format!("- Reviewed at: {}", review.reviewed_at))?;
    output.line(&format!(
        "- Duration: {}",
        format_duration(frozen.generated_at, review.reviewed_at)
    ))?;
    render_spec_context(output, &frozen.spec)?;
    render_frozen_context(output, frozen)?;
    output.fenced("Learner answer", &rated.submitted.response)?;
    render_ai_check(output, &review.result.check)
}

fn render_skipped_transcript_entry(
    output: &mut TranscriptWriter,
    skipped: &SkippedDrill,
) -> Result<(), TranscriptTooLarge> {
    output.line("")?;
    match skipped {
        SkippedDrill::Ready { frozen, skipped_at } => {
            output.line("- Status: skipped before answer")?;
            output.line(&format!("- Skipped at: {skipped_at}"))?;
            output.line(&format!(
                "- Duration: {}",
                format_duration(frozen.generated_at, *skipped_at)
            ))?;
            output.line("- Learner answer: not submitted")?;
            output.line("- Self-rating: not submitted")?;
            output.line("- Effective rating: none")?;
            output.line("- AI check: not run")?;
            render_spec_context(output, &frozen.spec)?;
            render_frozen_context(output, frozen)
        }
        SkippedDrill::Unresolved { result, skipped_at } => {
            let rated = &result.rated;
            let frozen = &rated.submitted.frozen;
            output.line("- Status: skipped after unresolved check")?;
            output.line(&format!("- Skipped at: {skipped_at}"))?;
            output.line(&format!(
                "- Duration: {}",
                format_duration(frozen.generated_at, *skipped_at)
            ))?;
            output.line(&format!("- Self-rating: {}", rated.learner_grade.as_str()))?;
            output.line("- Effective rating: none")?;
            output.line("- Override: no")?;
            render_spec_context(output, &frozen.spec)?;
            render_frozen_context(output, frozen)?;
            output.fenced("Learner answer", &rated.submitted.response)?;
            render_ai_check(output, &result.check)
        }
        SkippedDrill::GenerationFailed { spec, skipped_at } => {
            output.line("- Status: skipped after generation failure")?;
            output.line(&format!("- Skipped at: {skipped_at}"))?;
            output.line("- Question: unavailable (generation failed)")?;
            output.line("- Learner answer: not submitted")?;
            output.line("- Self-rating: not submitted")?;
            output.line("- Effective rating: none")?;
            output.line("- AI check: not run")?;
            render_spec_context(output, spec)
        }
    }
}

fn render_spec_context(
    output: &mut TranscriptWriter,
    spec: &DrillSpec,
) -> Result<(), TranscriptTooLarge> {
    output.fenced("Deck", &spec.deck_name)?;
    if let Some(goal) = &spec.goal {
        output.fenced("Goal", goal)?;
    }
    if let Some(source) = &spec.source {
        output.fenced("Source", source)?;
    }
    Ok(())
}

fn render_frozen_context(
    output: &mut TranscriptWriter,
    frozen: &FrozenQuestion,
) -> Result<(), TranscriptTooLarge> {
    output.line(&format!("- Generated at: {}", frozen.generated_at))?;
    output.line(&format!(
        "- Generation protocol version: {}",
        frozen.instance.protocol_version
    ))?;
    output.fenced("Generation model", &frozen.instance.model)?;
    output.fenced("Question", &frozen.instance.question)?;
    output.fenced("Target", &frozen.instance.target)?;
    output.fenced("Criteria", &frozen.instance.rubric)
}

fn render_ai_check(
    output: &mut TranscriptWriter,
    check: &CheckResult,
) -> Result<(), TranscriptTooLarge> {
    match check {
        CheckResult::Skipped => output.line("- AI check: skipped (Forgot self-rating)"),
        CheckResult::Evaluation(evaluation) => {
            output.line(&format!("- AI verdict: {}", evaluation.verdict.as_str()))?;
            output.line(&format!(
                "- AI protocol version: {}",
                evaluation.protocol_version
            ))?;
            output.fenced("AI model", &evaluation.model)?;
            output.fenced("AI comment", &evaluation.feedback)
        }
        CheckResult::Error(_) => output.line("- AI check: unavailable"),
    }
}

fn review_resolution_label(resolution: ReviewResolution) -> &'static str {
    match resolution {
        ReviewResolution::LearnerForgot => "learner forgot",
        ReviewResolution::AiConfirmed => "AI confirmed",
        ReviewResolution::AiRejected => "AI rejected",
        ReviewResolution::UserOverride => "user override",
        ReviewResolution::UndoRegrade => "undo regrade",
        ReviewResolution::LegacyV1 => "legacy review",
    }
}

async fn katex_css() -> impl IntoResponse {
    let (status, headers, bytes) = static_assets::katex_css_handler().await;
    let css = String::from_utf8_lossy(bytes).replace("/assets/katex/fonts/", "fonts/");
    (status, headers, css)
}

fn load_macros(collection_root: &Path) -> HashMap<String, String> {
    let Some(content) = read_macros_file(collection_root) else {
        return HashMap::new();
    };
    content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('%') {
                return None;
            }
            let (name, definition) = line.split_once(char::is_whitespace)?;
            let definition = definition.trim();
            (valid_macro_name(name) && valid_macro_definition(definition))
                .then(|| (name.to_string(), definition.to_string()))
        })
        .take(MAX_MACROS)
        .collect()
}

fn read_macros_file(collection_root: &Path) -> Option<String> {
    let canonical_root = std::fs::canonicalize(collection_root).ok()?;
    if !std::fs::metadata(&canonical_root).ok()?.is_dir() {
        return None;
    }

    // Build from the canonical root so no caller-provided symlink component is
    // retained. macros.tex itself must not be a symlink, even if its target
    // would remain inside the collection.
    let path = canonical_root.join("macros.tex");
    let link_metadata = std::fs::symlink_metadata(&path).ok()?;
    if link_metadata.file_type().is_symlink() || !link_metadata.is_file() {
        return None;
    }

    let canonical_path = std::fs::canonicalize(&path).ok()?;
    if canonical_path.parent() != Some(canonical_root.as_path()) {
        return None;
    }
    let metadata = std::fs::metadata(&canonical_path).ok()?;
    if !metadata.is_file() || metadata.len() > MAX_MACROS_BYTES {
        return None;
    }

    // Re-check the resolved path immediately before opening. Reading through
    // the canonical path and bounding the reader also protects against a file
    // that grows after the metadata check.
    let resolved_metadata = std::fs::symlink_metadata(&canonical_path).ok()?;
    if resolved_metadata.file_type().is_symlink() || !resolved_metadata.is_file() {
        return None;
    }
    let file = std::fs::File::open(&canonical_path).ok()?;
    let opened_metadata = file.metadata().ok()?;
    if !opened_metadata.is_file()
        || !same_file_identity(&link_metadata, &opened_metadata)
        || !same_file_identity(&resolved_metadata, &opened_metadata)
    {
        return None;
    }
    let current_metadata = std::fs::symlink_metadata(&canonical_path).ok()?;
    if current_metadata.file_type().is_symlink()
        || !current_metadata.is_file()
        || !same_file_identity(&current_metadata, &opened_metadata)
    {
        return None;
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_MACROS_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_MACROS_BYTES {
        return None;
    }
    String::from_utf8(bytes).ok()
}

#[cfg(unix)]
fn same_file_identity(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    // `std` does not expose a portable file identifier. The checks before and
    // after opening still reject stable link swaps on non-Unix platforms.
    left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
        && left.created().ok() == right.created().ok()
}

fn valid_macro_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    (2..=MAX_MACRO_NAME_BYTES).contains(&bytes.len())
        && bytes[0] == b'\\'
        && bytes[1..].iter().all(u8::is_ascii_alphabetic)
}

fn valid_macro_definition(definition: &str) -> bool {
    !definition.is_empty()
        && definition.len() <= MAX_MACRO_DEFINITION_BYTES
        && !definition.chars().any(char::is_control)
}

async fn macros_json(State(state): State<AppState>) -> impl IntoResponse {
    let macros = serde_json::to_string(&*state.macros).unwrap_or_else(|_| "{}".to_string());
    (
        [
            (CONTENT_TYPE, "application/json; charset=utf-8"),
            (CACHE_CONTROL, "private, no-store"),
            (X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (
                axum::http::header::HeaderName::from_static("cross-origin-resource-policy"),
                "same-origin",
            ),
        ],
        macros,
    )
}

async fn media_file(
    State(state): State<AppState>,
    AxumPath(path): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    let file = match validate_media_request(&state.collection_root, &path) {
        Ok(file) => file,
        Err(_) => {
            return (
                StatusCode::NOT_FOUND,
                [(CONTENT_TYPE, "text/plain; charset=utf-8")],
                "Not Found",
            )
                .into_response();
        }
    };
    serve_media(file, &headers).await
}

async fn root(State(state): State<AppState>) -> Response {
    if let Err(error) = ensure_current(&state).await {
        return error_page(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string());
    }
    let should_finish = {
        let mutable = state.mutable.lock().expect("session mutex poisoned");
        mutable.queue.is_empty()
            && mutable.current.is_none()
            && !mutable.finish_recorded
            && !mutable.finish_in_progress
            && mutable.finish_error.is_none()
    };
    if should_finish {
        let _ = finish_session(&state);
    }
    let mutable = state.mutable.lock().expect("session mutex poisoned");
    Html(
        page(
            &mutable,
            &state.collection_root,
            state.access.csrf_token(),
            state.access.base_path(),
        )
        .into_string(),
    )
    .into_response()
}

async fn retry_finish(State(state): State<AppState>, Form(form): Form<SessionForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    {
        let mutable = state.mutable.lock().expect("session mutex poisoned");
        if mutable.session_id != form.session_id
            || !mutable.queue.is_empty()
            || mutable.current.is_some()
            || mutable.finish_recorded
            || mutable.finish_in_progress
            || mutable.finish_error.is_none()
        {
            return stale_submission();
        }
    }
    let _ = finish_session(&state);
    Redirect::to("/").into_response()
}

async fn shutdown(State(state): State<AppState>, Form(form): Form<SessionForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    let sender = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if mutable.session_id == form.session_id && mutable.shutdown_requested {
            return Html(
                page(
                    &mutable,
                    &state.collection_root,
                    state.access.csrf_token(),
                    state.access.base_path(),
                )
                .into_string(),
            )
            .into_response();
        }
        if mutable.session_id != form.session_id
            || !mutable.finish_recorded
            || mutable.finished_at.is_none()
            || !mutable.queue.is_empty()
            || mutable.current.is_some()
            || mutable.suspended_current.is_some()
            || mutable.undo_in_progress
            || mutable.finish_in_progress
            || mutable.finish_error.is_some()
        {
            return stale_submission();
        }
        let Some(sender) = state
            .shutdown_tx
            .lock()
            .expect("shutdown mutex poisoned")
            .take()
        else {
            return error_page(
                StatusCode::INTERNAL_SERVER_ERROR,
                "The shutdown signal is not available.",
            );
        };
        mutable.shutdown_requested = true;
        sender
    };

    if let Err(error) = archive_session(&state) {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        mutable.shutdown_requested = false;
        *state.shutdown_tx.lock().expect("shutdown mutex poisoned") = Some(sender);
        return error_page(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Could not save generated drills before shutdown: {error}"),
        );
    }

    if sender.send(()).is_err() {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        mutable.shutdown_requested = false;
        return error_page(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The server could not receive the shutdown signal.",
        );
    }
    let mutable = state.mutable.lock().expect("session mutex poisoned");
    Html(
        page(
            &mutable,
            &state.collection_root,
            state.access.csrf_token(),
            state.access.base_path(),
        )
        .into_string(),
    )
    .into_response()
}

/// Generate and durably freeze the next instance before allowing the browser
/// to see its question.
async fn ensure_current(state: &AppState) -> Fallible<()> {
    let generation = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if mutable.undo_in_progress || mutable.current.is_some() || mutable.queue.is_empty() {
            return Ok(());
        }
        let spec = mutable.queue.front().expect("non-empty queue").clone();
        let token = mutable.next_token;
        mutable.next_token += 1;
        mutable.current = Some(Current::Generating {
            token,
            spec: spec.clone(),
        });
        (token, spec, mutable.session_id)
    };

    let (token, spec, session_id) = generation;
    let worker_state = state.clone();
    let worker_spec = spec.clone();
    let worker = tokio::spawn(async move {
        generate_current(worker_state, token, worker_spec, session_id).await;
    });
    if let Err(error) = worker.await {
        log::error!("Generation worker failed for token {token}: {error}");
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if current_token(&mutable.current) == Some(token) {
            mutable.current = Some(Current::GenerationFailed {
                token,
                spec,
                message: "Generation worker failed. Check the server log.".into(),
            });
        }
    }
    Ok(())
}

async fn generate_current(state: AppState, token: u64, spec: DrillSpec, session_id: i64) {
    // This can involve a subprocess and must remain outside every lock.
    let generated = match state.backend.generate(&spec).await {
        Ok(instance) => instance,
        Err(error) => {
            log::error!("Could not generate drill token {token}: {error}");
            let mut mutable = state.mutable.lock().expect("session mutex poisoned");
            if current_token(&mutable.current) == Some(token) {
                mutable.current = Some(Current::GenerationFailed {
                    token,
                    spec,
                    message: "Generation failed. Check the server log.".into(),
                });
            }
            return;
        }
    };

    // Freeze before rendering. This lock is independent of the session lock.
    let generated_at = Timestamp::now();
    let instance_id =
        match state
            .recorder
            .freeze_instance(session_id, &spec, &generated, generated_at)
        {
            Ok(instance_id) => instance_id,
            Err(error) => {
                log::error!("Could not freeze generated drill token {token}: {error}");
                let mut mutable = state.mutable.lock().expect("session mutex poisoned");
                if current_token(&mutable.current) == Some(token) {
                    mutable.current = Some(Current::GenerationFailed {
                        token,
                        spec,
                        message: "Could not freeze the generated drill. Check the server log."
                            .into(),
                    });
                }
                return;
            }
        };

    let mut mutable = state.mutable.lock().expect("session mutex poisoned");
    if current_token(&mutable.current) == Some(token) {
        mutable.current = Some(Current::Ready {
            frozen: FrozenQuestion {
                token,
                spec,
                instance_id,
                generated_at,
                instance: generated,
            },
            draft: String::new(),
            notice: None,
        });
    }
}

async fn answer(State(state): State<AppState>, Form(form): Form<AnswerForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    if form.response.trim().is_empty() {
        return reject("Write a response before submitting it.");
    }

    {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if mutable.undo_in_progress {
            return stale_submission();
        }
        let Some(Current::Ready { frozen, .. }) = mutable.current.as_ref() else {
            return stale_submission();
        };
        if mutable.session_id != form.session_id
            || frozen.token != form.token
            || frozen.instance_id != form.instance_id
        {
            return stale_submission();
        }
        mutable.current = Some(Current::Rating(SubmittedAnswer {
            frozen: frozen.clone(),
            response: form.response.clone(),
            notice: None,
        }));
    }
    Redirect::to("/").into_response()
}

async fn back(State(state): State<AppState>, Form(form): Form<TokenForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    let mut mutable = state.mutable.lock().expect("session mutex poisoned");
    let Some(Current::Rating(submitted)) = mutable.current.as_ref() else {
        return stale_submission();
    };
    if mutable.session_id != form.session_id
        || submitted.frozen.token != form.token
        || Some(submitted.frozen.instance_id) != form.instance_id
    {
        return stale_submission();
    }
    let mut submitted = submitted.clone();
    submitted.frozen.token = mutable.next_token;
    mutable.next_token += 1;
    mutable.current = Some(Current::Ready {
        frozen: submitted.frozen,
        draft: submitted.response,
        notice: None,
    });
    Redirect::to("/").into_response()
}

async fn rate(State(state): State<AppState>, Form(form): Form<RateForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    let learner_grade = match parse_grade(&form.grade) {
        Some(grade) => grade,
        None => return reject("Unknown scheduler grade."),
    };
    let submitted = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        let Some(Current::Rating(submitted)) = mutable.current.as_ref() else {
            return stale_submission();
        };
        if mutable.session_id != form.session_id
            || submitted.frozen.token != form.token
            || submitted.frozen.instance_id != form.instance_id
        {
            return stale_submission();
        }
        let submitted = submitted.clone();
        mutable.current = Some(Current::Recording {
            submitted: submitted.clone(),
            grade: learner_grade,
        });
        submitted
    };

    // The first response and its self-rating are durable before an optional
    // model call begins. Forgot therefore never needs an evaluator result.
    let attempt_id = match state.recorder.record_attempt(
        submitted.frozen.instance_id,
        &submitted.response,
        learner_grade,
        Timestamp::now(),
    ) {
        Ok(attempt_id) => attempt_id,
        Err(error) => {
            let mut submitted = submitted;
            submitted.notice = Some(format!("Could not record the answer: {error}"));
            let mut mutable = state.mutable.lock().expect("session mutex poisoned");
            if current_token(&mutable.current) == Some(form.token) {
                mutable.current = Some(Current::Rating(submitted));
            }
            return Redirect::to("/").into_response();
        }
    };

    let rated = RatedAnswer {
        submitted,
        learner_grade,
        attempt_id,
    };
    if learner_grade == Grade::Forgot {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if current_token(&mutable.current) == Some(form.token) {
            mutable.current = Some(Current::Result(ReviewResult {
                rated,
                check: CheckResult::Skipped,
                default_grade: Some(Grade::Forgot),
                notice: None,
                regrading: false,
            }));
        }
        return Redirect::to("/").into_response();
    }

    {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if current_token(&mutable.current) == Some(form.token) {
            mutable.current = Some(Current::Evaluating {
                rated: rated.clone(),
            });
        }
    }
    complete_evaluation(&state, rated).await;
    Redirect::to("/").into_response()
}

async fn evaluate_rated(state: &AppState, rated: RatedAnswer) -> ReviewResult {
    let evaluation = match state
        .backend
        .evaluate(&rated.submitted.frozen.instance, &rated.submitted.response)
        .await
    {
        Ok(evaluation) => evaluation,
        Err(error) => {
            return ReviewResult {
                rated,
                check: CheckResult::Error(format!("AI check failed: {error}")),
                default_grade: None,
                notice: None,
                regrading: false,
            };
        }
    };

    {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if current_token(&mutable.current) == Some(rated.submitted.frozen.token) {
            mutable.current = Some(Current::RecordingEvaluation {
                rated: rated.clone(),
                evaluation: evaluation.clone(),
            });
        }
    }
    if let Err(error) =
        state
            .recorder
            .attach_evaluation(rated.attempt_id, &evaluation, Timestamp::now())
    {
        return ReviewResult {
            rated,
            check: CheckResult::Error(format!("Could not record the AI check: {error}")),
            default_grade: None,
            notice: None,
            regrading: false,
        };
    }

    let default_grade = match evaluation.verdict {
        Verdict::Pass => Some(rated.learner_grade),
        Verdict::Partial | Verdict::Fail => Some(Grade::Forgot),
        Verdict::Uncertain | Verdict::Invalid => None,
    };
    ReviewResult {
        rated,
        check: CheckResult::Evaluation(evaluation),
        default_grade,
        notice: None,
        regrading: false,
    }
}

async fn complete_evaluation(state: &AppState, rated: RatedAnswer) {
    let token = rated.submitted.frozen.token;
    let worker_state = state.clone();
    let failed_rated = rated.clone();
    let worker = tokio::spawn(async move {
        let result = evaluate_rated(&worker_state, rated).await;
        let mut mutable = worker_state.mutable.lock().expect("session mutex poisoned");
        if current_token(&mutable.current) == Some(token) {
            mutable.current = Some(Current::Result(result));
        }
    });
    if let Err(error) = worker.await {
        log::error!("Evaluation worker failed for token {token}: {error}");
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if current_token(&mutable.current) == Some(token) {
            mutable.current = Some(Current::Result(ReviewResult {
                rated: failed_rated,
                check: CheckResult::Error("AI check worker failed. Check the server log.".into()),
                default_grade: None,
                notice: None,
                regrading: false,
            }));
        }
    }
}

async fn retry_evaluation(State(state): State<AppState>, Form(form): Form<TokenForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    let rated = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        let Some(Current::Result(result)) = mutable.current.as_ref() else {
            return stale_submission();
        };
        if mutable.session_id != form.session_id
            || result.rated.submitted.frozen.token != form.token
            || Some(result.rated.submitted.frozen.instance_id) != form.instance_id
            || !matches!(result.check, CheckResult::Error(_))
            || result.rated.learner_grade == Grade::Forgot
        {
            return stale_submission();
        }
        let rated = result.rated.clone();
        mutable.current = Some(Current::Evaluating {
            rated: rated.clone(),
        });
        rated
    };
    complete_evaluation(&state, rated).await;
    Redirect::to("/").into_response()
}

async fn accept(State(state): State<AppState>, Form(form): Form<AcceptForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    let (mut result, grade, resolution) = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        let Some(Current::Result(result)) = mutable.current.as_ref() else {
            return stale_submission();
        };
        if mutable.session_id != form.session_id
            || result.rated.submitted.frozen.token != form.token
            || result.rated.submitted.frozen.instance_id != form.instance_id
        {
            return stale_submission();
        }
        let Some((grade, resolution)) = accepted_choice(result, &form) else {
            return reject("That grade is not available for this result.");
        };
        let result = result.clone();
        mutable.current = Some(Current::Saving {
            result: result.clone(),
            grade,
        });
        (result, grade, resolution)
    };

    let reviewed_at = Timestamp::now();
    match state
        .recorder
        .accept_grade(result.rated.attempt_id, grade, resolution, reviewed_at)
    {
        Ok(review_id) => {
            result.notice = None;
            result.regrading = false;
            let completed = CompletedReview {
                result,
                review_id,
                reviewed_at,
                accepted_grade: grade,
                resolution,
            };
            let finished = advance_completed(&state, form.token, completed);
            if finished {
                let _ = finish_session(&state);
            }
        }
        Err(error) => {
            result.notice = Some(format!("Could not save the grade: {error}"));
            let mut mutable = state.mutable.lock().expect("session mutex poisoned");
            if current_token(&mutable.current) == Some(form.token) {
                mutable.current = Some(Current::Result(result));
            }
        }
    }
    Redirect::to("/").into_response()
}

fn accepted_choice(result: &ReviewResult, form: &AcceptForm) -> Option<(Grade, ReviewResolution)> {
    match form.action.as_str() {
        "default" if !result.regrading => {
            let grade = result.default_grade?;
            let resolution = match &result.check {
                CheckResult::Skipped => ReviewResolution::LearnerForgot,
                CheckResult::Evaluation(evaluation) if evaluation.verdict == Verdict::Pass => {
                    ReviewResolution::AiConfirmed
                }
                CheckResult::Evaluation(evaluation)
                    if matches!(evaluation.verdict, Verdict::Partial | Verdict::Fail) =>
                {
                    ReviewResolution::AiRejected
                }
                _ => return None,
            };
            Some((grade, resolution))
        }
        "override" if !result.regrading && result_is_nay(result) => {
            let grade = parse_grade(form.grade.as_deref()?)?;
            (grade == result.rated.learner_grade).then_some((grade, ReviewResolution::UserOverride))
        }
        "regrade" if result.regrading => {
            let grade = parse_grade(form.grade.as_deref()?)?;
            Some((grade, ReviewResolution::UndoRegrade))
        }
        _ => None,
    }
}

fn result_is_nay(result: &ReviewResult) -> bool {
    matches!(
        &result.check,
        CheckResult::Evaluation(Evaluation {
            verdict: Verdict::Partial | Verdict::Fail,
            ..
        })
    )
}

async fn undo(State(state): State<AppState>, Form(form): Form<UndoForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    let (completed, undo_token, replace_forgot) = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        let Some(completed) = mutable.completed_reviews.last() else {
            return reject("There is no completed review to undo.");
        };
        if mutable.session_id != form.session_id
            || completed.review_id != form.review_id
            || !mutable.transcript_entries.iter().any(|entry| {
                matches!(
                    entry,
                    SessionTranscriptEntry::Reviewed(review)
                        if review.review_id == completed.review_id
                )
            })
            || mutable.undo_in_progress
            || mutable.finish_in_progress
            || mutable.shutdown_requested
            || mutable.archive_publication.locks_undo()
            || mutable.suspended_current.is_some()
            || !undo_is_safe(&mutable.current)
        {
            return stale_submission();
        }
        let completed = completed.clone();
        if let Some(Current::Ready { frozen, draft, .. }) = mutable.current.as_mut() {
            if form.current_token != Some(frozen.token)
                || form.current_instance_id != Some(frozen.instance_id)
            {
                return stale_submission();
            }
            let Some(current_draft) = form.current_draft.as_ref() else {
                return stale_submission();
            };
            draft.clone_from(current_draft);
        }
        let replace_forgot = matches!(completed.result.check, CheckResult::Skipped);
        let undo_token = mutable.next_token;
        mutable.next_token += 1;
        mutable.undo_in_progress = true;
        (completed, undo_token, replace_forgot)
    };

    if let Err(error) = state
        .recorder
        .undo_review(completed.review_id, Timestamp::now())
    {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        mutable.undo_in_progress = false;
        log::error!("Could not undo review {}: {error}", completed.review_id);
        return error_page(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Could not undo the previous review: {error}"),
        );
    }

    // A Forgot attempt deliberately has no AI evaluation. If it is undone,
    // duplicate the frozen instance and collect a fresh self-rating/attempt;
    // direct success regrades would otherwise have no correctness check.
    let replacement = replace_forgot.then(|| {
        let generated_at = Timestamp::now();
        state
            .recorder
            .freeze_instance(
                form.session_id,
                &completed.result.rated.submitted.frozen.spec,
                &completed.result.rated.submitted.frozen.instance,
                generated_at,
            )
            .map(|instance_id| FrozenQuestion {
                token: undo_token,
                spec: completed.result.rated.submitted.frozen.spec.clone(),
                instance_id,
                generated_at,
                instance: completed.result.rated.submitted.frozen.instance.clone(),
            })
    });

    let mut mutable = state.mutable.lock().expect("session mutex poisoned");
    let Some(last) = mutable.completed_reviews.last() else {
        mutable.undo_in_progress = false;
        return stale_submission();
    };
    if last.review_id != completed.review_id {
        mutable.undo_in_progress = false;
        return stale_submission();
    }
    mutable.completed_reviews.pop();
    mutable.transcript_entries.retain(|entry| {
        !matches!(
            entry,
            SessionTranscriptEntry::Reviewed(review)
                if review.review_id == completed.review_id
        )
    });
    mutable.completed = mutable.completed.saturating_sub(1);
    mutable
        .queue
        .push_front(completed.result.rated.submitted.frozen.spec.clone());
    mutable.suspended_current = mutable.current.take();
    let mut result = completed.result;
    result.rated.submitted.frozen.token = undo_token;
    mutable.current = match replacement {
        Some(Ok(frozen)) => Some(Current::Rating(SubmittedAnswer {
            frozen,
            response: result.rated.submitted.response,
            notice: Some("Previous review undone. Rate the answer again.".into()),
        })),
        Some(Err(error)) => {
            log::error!("Could not freeze replacement after undo: {error}");
            result.default_grade = None;
            result.notice = Some(format!(
                "Review undone, but a checked replacement could not be prepared: {error}"
            ));
            result.regrading = true;
            Some(Current::Result(result))
        }
        None => {
            result.default_grade = None;
            result.notice = Some("Previous review undone. Select the replacement grade.".into());
            result.regrading = true;
            Some(Current::Result(result))
        }
    };
    mutable.finish_recorded = false;
    mutable.finished_at = None;
    mutable.finish_error = None;
    mutable.completion_history = None;
    mutable.history_error = None;
    mutable.archive_error = None;
    mutable.undo_in_progress = false;
    Redirect::to("/").into_response()
}

fn undo_is_safe(current: &Option<Current>) -> bool {
    matches!(
        current,
        None | Some(Current::Ready { .. }) | Some(Current::GenerationFailed { .. })
    )
}

async fn regenerate(State(state): State<AppState>, Form(form): Form<TokenForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        let allowed = !mutable.undo_in_progress
            && mutable.session_id == form.session_id
            && match mutable.current.as_ref() {
                Some(Current::Result(result)) => {
                    result.rated.submitted.frozen.token == form.token
                        && Some(result.rated.submitted.frozen.instance_id) == form.instance_id
                        && result_cannot_schedule(result)
                }
                Some(Current::GenerationFailed { token, .. }) => {
                    *token == form.token && form.instance_id.is_none()
                }
                _ => false,
            };
        if !allowed {
            return stale_submission();
        }
        mutable.current = None;
    }
    if let Err(error) = ensure_current(&state).await {
        return error_page(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string());
    }
    Redirect::to("/").into_response()
}

async fn skip(State(state): State<AppState>, Form(form): Form<TokenForm>) -> Response {
    if !state.access.csrf_is_valid(&form.csrf_token) {
        return security_error(StatusCode::FORBIDDEN, "Forbidden");
    }
    // Validate the exact phase and advance under one lock. A concurrent answer
    // or grade cannot move this token into an in-flight phase between check
    // and removal.
    let finished = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if mutable.undo_in_progress || mutable.session_id != form.session_id {
            return stale_submission();
        }
        let skipped_at = Timestamp::now();
        let skipped = match mutable.current.as_ref() {
            Some(Current::Ready { frozen, .. })
                if frozen.token == form.token && Some(frozen.instance_id) == form.instance_id =>
            {
                SkippedDrill::Ready {
                    frozen: frozen.clone(),
                    skipped_at,
                }
            }
            Some(Current::Result(result))
                if result.rated.submitted.frozen.token == form.token
                    && Some(result.rated.submitted.frozen.instance_id) == form.instance_id
                    && !result.regrading
                    && result_cannot_schedule(result) =>
            {
                SkippedDrill::Unresolved {
                    result: result.clone(),
                    skipped_at,
                }
            }
            Some(Current::GenerationFailed { token, spec, .. })
                if *token == form.token && form.instance_id.is_none() =>
            {
                SkippedDrill::GenerationFailed {
                    spec: spec.clone(),
                    skipped_at,
                }
            }
            _ => return stale_submission(),
        };
        mutable.current = mutable.suspended_current.take();
        mutable.queue.pop_front();
        mutable.completed += 1;
        mutable
            .transcript_entries
            .push(SessionTranscriptEntry::Skipped(skipped));
        mutable.queue.is_empty()
    };
    if finished {
        let _ = finish_session(&state);
    }
    Redirect::to("/").into_response()
}

fn advance_completed(state: &AppState, token: u64, completed: CompletedReview) -> bool {
    let mut mutable = state.mutable.lock().expect("session mutex poisoned");
    if current_token(&mutable.current) != Some(token) {
        return false;
    }
    mutable.current = mutable.suspended_current.take();
    mutable.queue.pop_front();
    mutable.completed += 1;
    mutable
        .transcript_entries
        .push(SessionTranscriptEntry::Reviewed(completed.clone()));
    mutable.completed_reviews.push(completed);
    mutable.queue.is_empty()
}

fn finish_session(state: &AppState) -> Fallible<()> {
    finish_session_inner(state, false)
}

fn finish_session_on_shutdown(state: &AppState) -> Fallible<()> {
    finish_session_inner(state, true)?;
    archive_session(state)
}

fn finish_session_inner(state: &AppState, force: bool) -> Fallible<()> {
    let session_id = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        if mutable.finish_recorded
            || mutable.finish_in_progress
            || mutable.shutdown_requested
            || (!force
                && (mutable.undo_in_progress
                    || !mutable.queue.is_empty()
                    || mutable.current.is_some()))
        {
            return Ok(());
        }
        mutable.finish_in_progress = true;
        mutable.finish_error = None;
        mutable.session_id
    };
    let finished_at = Timestamp::now();
    let result = state.recorder.finish_session(session_id, finished_at);
    let history = if result.is_ok() {
        Some(
            state
                .recorder
                .completion_history_for_specs(Date::today(), &state.specs),
        )
    } else {
        None
    };
    let mut mutable = state.mutable.lock().expect("session mutex poisoned");
    mutable.finish_in_progress = false;
    match result {
        Ok(()) => {
            mutable.finish_recorded = true;
            mutable.finished_at = Some(finished_at);
            mutable.finish_error = None;
            match history.expect("history query is present after finalization") {
                Ok(history) => {
                    mutable.completion_history = Some(history);
                    mutable.history_error = None;
                }
                Err(error) => {
                    let message = format!("History unavailable: {error}");
                    log::error!("{message}");
                    mutable.completion_history = None;
                    mutable.history_error = Some(message);
                }
            }
            Ok(())
        }
        Err(error) => {
            let message = format!("Could not finalize session: {error}");
            log::error!("{message}");
            mutable.finish_error = Some(message);
            Err(error)
        }
    }
}

fn archive_session(state: &AppState) -> Fallible<()> {
    let Some(output_dir) = state.archive_dir.as_deref() else {
        return Ok(());
    };
    let (session_id, completed_reviews) = {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        match mutable.archive_publication {
            ArchivePublication::Published => return Ok(()),
            ArchivePublication::Publishing => {
                return Err(crate::error::ErrorReport::new(
                    "generated drill publication is already in progress",
                ));
            }
            ArchivePublication::NotStarted | ArchivePublication::Failed => {}
        }
        // From this point onward an immutable archive file may exist, even if
        // a later write in the same batch fails. Seal Undo before the first
        // filesystem operation so the database cannot diverge from a
        // partially published batch.
        mutable.archive_publication = ArchivePublication::Publishing;
        mutable.archive_error = None;
        (mutable.session_id, mutable.completed_reviews.clone())
    };
    if let Some(message) = archive_completed_reviews(output_dir, session_id, &completed_reviews) {
        let mut mutable = state.mutable.lock().expect("session mutex poisoned");
        mutable.archive_publication = ArchivePublication::Failed;
        mutable.archive_error = Some(message.clone());
        return Err(crate::error::ErrorReport::new(message));
    }
    let mut mutable = state.mutable.lock().expect("session mutex poisoned");
    mutable.archive_publication = ArchivePublication::Published;
    mutable.archive_error = None;
    Ok(())
}

fn archive_completed_reviews(
    output_dir: &Path,
    session_id: i64,
    completed_reviews: &[CompletedReview],
) -> Option<String> {
    for completed in completed_reviews {
        let frozen = &completed.result.rated.submitted.frozen;
        if frozen.spec.question.directive_count() == 0 && frozen.spec.answer.directive_count() == 0
        {
            continue;
        }
        let record = ArchivedDrill {
            spec: &frozen.spec,
            instance: &frozen.instance,
            generated_at: frozen.generated_at,
            reviewed_at: completed.reviewed_at,
            trace: ArchiveTrace {
                session_id,
                instance_id: frozen.instance_id,
                attempt_id: completed.result.rated.attempt_id,
                review_id: completed.review_id,
            },
        };
        match save_generated_drill(output_dir, &record) {
            Ok(saved) => log::info!("Saved generated drill to {}", saved.path.display()),
            Err(error) => {
                let message = format!("Could not save a generated drill: {error}");
                log::error!("{message}");
                return Some(message);
            }
        }
    }
    None
}

fn current_token(current: &Option<Current>) -> Option<u64> {
    match current.as_ref()? {
        Current::Generating { token, .. } | Current::GenerationFailed { token, .. } => Some(*token),
        Current::Ready { frozen, .. } => Some(frozen.token),
        Current::Rating(submitted) | Current::Recording { submitted, .. } => {
            Some(submitted.frozen.token)
        }
        Current::Evaluating { rated } | Current::RecordingEvaluation { rated, .. } => {
            Some(rated.submitted.frozen.token)
        }
        Current::Result(result) | Current::Saving { result, .. } => {
            Some(result.rated.submitted.frozen.token)
        }
    }
}

fn parse_grade(value: &str) -> Option<Grade> {
    match value {
        "forgot" => Some(Grade::Forgot),
        "hard" => Some(Grade::Hard),
        "good" => Some(Grade::Good),
        "easy" => Some(Grade::Easy),
        _ => None,
    }
}

fn result_cannot_schedule(result: &ReviewResult) -> bool {
    matches!(&result.check, CheckResult::Error(_))
        || matches!(
            &result.check,
            CheckResult::Evaluation(Evaluation {
                verdict: Verdict::Uncertain | Verdict::Invalid,
                ..
            })
        )
}

/// Build the queue using only hashes present in the current parse. This keeps
/// historical/orphaned rows from reappearing in practice sessions.
fn build_queue(
    storage: &Storage,
    specs: &[DrillSpec],
    deck_filter: Option<&str>,
    due_limit: Option<usize>,
    new_limit: Option<usize>,
) -> Fallible<Vec<DrillSpec>> {
    let selected =
        storage.due_queue_for_specs(Date::today(), specs, deck_filter, new_limit, due_limit)?;
    let mut current: HashMap<SpecHash, DrillSpec> = specs
        .iter()
        .cloned()
        .map(|spec| (spec.hash(), spec))
        .collect();
    let mut queue = Vec::new();
    for row in selected {
        if let Some(spec) = current.remove(&row.spec_hash) {
            queue.push(spec);
        }
    }
    Ok(queue)
}

fn relative_route(route: &str) -> &str {
    route.strip_prefix('/').unwrap_or(route)
}

fn render_markdown(
    markdown: &str,
    collection_root: &Path,
    spec_path: &Path,
    base_path: &str,
) -> String {
    let rendered = markdown_to_html(markdown, collection_root, spec_path);
    if base_path.is_empty() {
        rendered
    } else {
        rendered.replace("\"/file/", &format!("\"{base_path}/file/"))
    }
}

fn page(
    session: &SessionState,
    collection_root: &Path,
    csrf_token: &str,
    base_path: &str,
) -> Markup {
    let content = if session.shutdown_requested {
        shutdown_page()
    } else if session.queue.is_empty() && session.current.is_none() {
        if session.finish_recorded {
            finished_page(session, csrf_token)
        } else {
            finalization_page(session, csrf_token)
        }
    } else {
        practice_page(session, collection_root, csrf_token, base_path)
    };
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { "Hashdrills" }
                link rel="stylesheet" href=(relative_route(static_assets::KATEX_CSS_URL));
                link rel="stylesheet" href=(relative_route(static_assets::HIGHLIGHT_CSS_URL));
                script defer src=(relative_route(static_assets::KATEX_JS_URL)) {}
                script defer src=(relative_route(static_assets::KATEX_MHCHEM_JS_URL)) {}
                script defer src=(relative_route(static_assets::HIGHLIGHT_JS_URL)) {}
                style { (PreEscaped(STYLES)) }
            }
            body {
                @if let Some(message) = &session.archive_error {
                    div class="notice error global-notice" role="alert" { (message) }
                }
                (content)
                script { (PreEscaped(SCRIPT)) }
            }
        }
    }
}

fn practice_page(
    session: &SessionState,
    collection_root: &Path,
    csrf_token: &str,
    base_path: &str,
) -> Markup {
    let percent = session
        .completed
        .saturating_mul(100)
        .checked_div(session.total)
        .unwrap_or(100);
    html! {
        main {
            header {
                h1 class="brand" { "Hashdrills" }
                div class="progress-label" {
                    (session.completed) " / " (session.total)
                }
            }
            div class="progress" role="progressbar" aria-label="Session progress"
                aria-valuemin="0" aria-valuemax=(session.total) aria-valuenow=(session.completed) {
                div class="progress-fill" style=(format!("width: {percent}%")) {}
            }
            (match session.current.as_ref() {
                Some(Current::Generating { spec, .. }) => waiting_card(&spec.deck_name, "GENERATING"),
                Some(Current::Recording { submitted, grade }) => html! {
                    (waiting_card(&submitted.frozen.spec.deck_name, "RECORDING"))
                    div class="sr-only" { (&submitted.response) " " (grade.as_str()) }
                },
                Some(Current::Evaluating { rated }) => html! {
                    (waiting_card(&rated.submitted.frozen.spec.deck_name, "AI CHECK"))
                    div class="sr-only" { (&rated.submitted.response) }
                },
                Some(Current::RecordingEvaluation { rated, evaluation }) => html! {
                    (waiting_card(&rated.submitted.frozen.spec.deck_name, "RECORDING CHECK"))
                    div class="sr-only" { (evaluation.verdict.as_str()) }
                },
                Some(Current::Saving { result, grade }) => html! {
                    (waiting_card(&result.rated.submitted.frozen.spec.deck_name, "SAVING REVIEW"))
                    div class="sr-only" { (grade.as_str()) }
                },
                Some(Current::Ready { frozen, draft, notice }) => question_card(session.session_id, frozen, draft, notice.as_deref(), collection_root, csrf_token, base_path),
                Some(Current::Rating(submitted)) => rating_card(session.session_id, submitted, collection_root, csrf_token, base_path),
                Some(Current::Result(result)) => result_card(session.session_id, result, collection_root, csrf_token, base_path),
                Some(Current::GenerationFailed { token, spec, message }) => generation_error(session.session_id, *token, spec, message, csrf_token),
                None => waiting_card("", "PREPARING"),
            })
            (undo_control(session, csrf_token))
        }
    }
}

fn waiting_card(deck: &str, message: &str) -> Markup {
    html! {
        section class="card waiting" data-poll-current {
            @if !deck.is_empty() { div class="deck" { (deck) } }
            p class="status" { (message) }
        }
    }
}

fn deck_header(spec: &DrillSpec) -> Markup {
    html! {
        div class="deck-header" {
            div class="deck" { (&spec.deck_name) }
            @if let Some(source) = &spec.source {
                a class="source" href=(source) target="_blank" rel="noopener noreferrer"
                    title="Open the source for this deck." {
                    "Source ↗"
                }
            }
        }
    }
}

/// Render only the displayed question. Target, rubric, goal, and evaluator
/// context intentionally do not occur anywhere in this pre-answer response.
fn question_card(
    session_id: i64,
    frozen: &FrozenQuestion,
    draft: &str,
    notice: Option<&str>,
    collection_root: &Path,
    csrf_token: &str,
    base_path: &str,
) -> Markup {
    let question_classes = question_classes(&frozen.instance.question);
    html! {
        section class="card" {
            (deck_header(&frozen.spec))
            @if let Some(notice) = notice {
                div class="notice error" role="alert" { (notice) }
            }
            div class=(question_classes) {
                (PreEscaped(render_markdown(
                    &frozen.instance.question,
                    collection_root,
                    &frozen.spec.path,
                    base_path,
                )))
            }
            form action="answer" method="post" data-answer-form {
                (csrf_field(csrf_token))
                input type="hidden" name="session_id" value=(session_id);
                input type="hidden" name="token" value=(frozen.token);
                input type="hidden" name="instance_id" value=(frozen.instance_id);
                label class="sr-only" for="response" { "Answer" }
                textarea id="response" name="response" rows="8" placeholder="Answer" autofocus required { (draft) }
                p class="shortcut" { "SHIFT+ENTER newline" }
                div class="actions" {
                    button class="primary" type="submit" { "Submit " kbd aria-label="Enter" { "↵" } }
                    button class="quiet" type="submit" formaction="skip" formnovalidate { "Skip" }
                }
            }
        }
    }
}

fn question_classes(question: &str) -> &'static str {
    let visible_units = question
        .chars()
        .filter(|character| !character.is_whitespace())
        .count();
    match visible_units {
        0..=240 => "question rich-text",
        241..=500 => "question question-long rich-text",
        _ => "question question-very-long rich-text",
    }
}

fn rating_card(
    session_id: i64,
    submitted: &SubmittedAnswer,
    collection_root: &Path,
    csrf_token: &str,
    base_path: &str,
) -> Markup {
    let frozen = &submitted.frozen;
    html! {
        section class="card" {
            (deck_header(&frozen.spec))
            div class="question compact rich-text" {
                (PreEscaped(render_markdown(
                    &frozen.instance.question,
                    collection_root,
                    &frozen.spec.path,
                    base_path,
                )))
            }
            div class="response" aria-label="Answer" { (&submitted.response) }
            @if let Some(message) = &submitted.notice {
                div class="notice error" role="alert" { (message) }
            }
            div class="rating-sheet" role="dialog" aria-label="Rate answer" tabindex="-1" data-rating-focus {
                form action="rate" method="post" class="grades" data-rate-form {
                    (csrf_field(csrf_token))
                    input type="hidden" name="session_id" value=(session_id);
                    input type="hidden" name="token" value=(frozen.token);
                    input type="hidden" name="instance_id" value=(frozen.instance_id);
                    button type="submit" name="grade" value="forgot" { kbd { "1" } " Forgot" }
                    button type="submit" name="grade" value="hard" { kbd { "2" } " Hard" }
                    button type="submit" name="grade" value="good" { kbd { "3" } " Good" }
                    button type="submit" name="grade" value="easy" { kbd { "4" } " Easy" }
                }
                form action="back" method="post" class="edit-answer" data-back-form {
                    (csrf_field(csrf_token))
                    input type="hidden" name="session_id" value=(session_id);
                    input type="hidden" name="token" value=(frozen.token);
                    input type="hidden" name="instance_id" value=(frozen.instance_id);
                    button class="text-button" type="submit" { kbd { "U" } " Edit answer" }
                }
            }
        }
    }
}

fn result_card(
    session_id: i64,
    result: &ReviewResult,
    collection_root: &Path,
    csrf_token: &str,
    base_path: &str,
) -> Markup {
    let rated = &result.rated;
    let frozen = &rated.submitted.frozen;
    html! {
        section class="card" data-result-focus tabindex="-1" aria-labelledby="result-title" {
            h1 id="result-title" class="sr-only" { "Review result" }
            (deck_header(&frozen.spec))
            div class="question compact result-question rich-text" {
                (PreEscaped(render_markdown(
                    &frozen.instance.question,
                    collection_root,
                    &frozen.spec.path,
                    base_path,
                )))
            }
            @if let Some(goal) = &frozen.spec.goal {
                details class="goal" {
                    summary { "GOAL" }
                    div class="reference rich-text" {
                        (PreEscaped(render_markdown(goal, collection_root, &frozen.spec.path, base_path)))
                    }
                }
            }
            div class="result-grid" {
                div class="response" aria-label="Answer" { (&rated.submitted.response) }
                div class="datum" aria-label="Self-rating" { (rated.learner_grade.as_str()) }
            }
            (check_panel(result))
            h2 { "CRITERIA" }
            div class="reference rich-text" {
                (PreEscaped(render_markdown(
                    &frozen.instance.target,
                    collection_root,
                    &frozen.spec.path,
                    base_path,
                )))
            }
            @if frozen.instance.rubric != frozen.instance.target {
                h2 { "RUBRIC" }
                div class="reference rich-text" {
                    (PreEscaped(render_markdown(
                        &frozen.instance.rubric,
                        collection_root,
                        &frozen.spec.path,
                        base_path,
                    )))
                }
            }
            @if let Some(message) = &result.notice {
                div class="notice error" role="alert" { (message) }
            }
            (result_actions(session_id, result, csrf_token))
        }
    }
}

fn check_panel(result: &ReviewResult) -> Markup {
    match &result.check {
        CheckResult::Skipped => html! {
            div class="check" {
                strong { "FORGOT" }
                span { "AI check skipped" }
            }
        },
        CheckResult::Evaluation(evaluation) => html! {
            div class=(format!("check {}", verdict_class(evaluation.verdict))) {
                strong {
                    (check_label(evaluation.verdict))
                    @if let Some(icon) = check_icon(evaluation.verdict) {
                        " " span class="verdict-icon" aria-hidden="true" { (icon) }
                    }
                }
                @if !evaluation.feedback.is_empty() {
                    span { (&evaluation.feedback) }
                }
            }
        },
        CheckResult::Error(message) => html! {
            div class="check error" role="alert" {
                strong { "CHECK ERROR" }
                span { (message) }
            }
        },
    }
}

fn check_label(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Pass => "YEA",
        Verdict::Partial | Verdict::Fail => "NAY",
        Verdict::Uncertain => "UNCERTAIN",
        Verdict::Invalid => "INVALID",
    }
}

fn check_icon(verdict: Verdict) -> Option<&'static str> {
    match verdict {
        Verdict::Pass => Some("✓"),
        Verdict::Partial | Verdict::Fail => Some("×"),
        Verdict::Uncertain | Verdict::Invalid => None,
    }
}

fn result_actions(session_id: i64, result: &ReviewResult, csrf_token: &str) -> Markup {
    let frozen = &result.rated.submitted.frozen;
    if result.regrading {
        if matches!(&result.check, CheckResult::Skipped) {
            return html! {
                h2 { "REPLACEMENT GRADE" }
                form action="accept" method="post" class="actions" {
                    (result_hidden_fields(session_id, frozen, csrf_token))
                    input type="hidden" name="action" value="regrade";
                    button class="primary" type="submit" name="grade" value="forgot" { "Restore Forgot" }
                }
            };
        }
        return html! {
            h2 { "REPLACEMENT GRADE" }
            form action="accept" method="post" class="grades" data-regrade-form data-rating-focus tabindex="-1" {
                (result_hidden_fields(session_id, frozen, csrf_token))
                input type="hidden" name="action" value="regrade";
                button type="submit" name="grade" value="forgot" { kbd { "1" } " Forgot" }
                button type="submit" name="grade" value="hard" { kbd { "2" } " Hard" }
                button type="submit" name="grade" value="good" { kbd { "3" } " Good" }
                button type="submit" name="grade" value="easy" { kbd { "4" } " Easy" }
            }
        };
    }

    if result.default_grade.is_some() {
        return html! {
            div class="actions split result-actions" {
                @if result.notice.is_none() {
                    form action="accept" method="post" data-default-form data-decision-form {
                        (result_hidden_fields(session_id, frozen, csrf_token))
                        input type="hidden" name="action" value="default";
                        button class="primary" type="submit" {
                            @if result_is_nay(result) { "Continue as Forgot" } @else { "Continue" }
                        }
                    }
                } @else {
                    form action="accept" method="post" data-default-form data-decision-form {
                        (result_hidden_fields(session_id, frozen, csrf_token))
                        input type="hidden" name="action" value="default";
                        button class="primary" type="submit" { "Retry save" }
                    }
                }
                @if result_is_nay(result) {
                    form action="accept" method="post" data-decision-form data-override-form {
                        (result_hidden_fields(session_id, frozen, csrf_token))
                        input type="hidden" name="action" value="override";
                        input type="hidden" name="grade" value=(result.rated.learner_grade.as_str());
                        button class="quiet" type="submit" { "Keep my rating" }
                    }
                }
            }
        };
    }

    html! {
        div class="notice neutral" {
            "No schedule update."
        }
        div class="actions result-actions" {
            @if matches!(&result.check, CheckResult::Error(_)) {
                form action="retry-evaluation" method="post" {
                    (result_hidden_fields(session_id, frozen, csrf_token))
                    button class="primary" type="submit" { "Retry check" }
                }
            }
            form action="regenerate" method="post" {
                (result_hidden_fields(session_id, frozen, csrf_token))
                button class="quiet" type="submit" { "Regenerate" }
            }
            form action="skip" method="post" {
                (result_hidden_fields(session_id, frozen, csrf_token))
                button class="quiet" type="submit" { "Skip" }
            }
        }
    }
}

fn result_hidden_fields(session_id: i64, frozen: &FrozenQuestion, csrf_token: &str) -> Markup {
    html! {
        (csrf_field(csrf_token))
        input type="hidden" name="session_id" value=(session_id);
        input type="hidden" name="token" value=(frozen.token);
        input type="hidden" name="instance_id" value=(frozen.instance_id);
    }
}

fn undo_control(session: &SessionState, csrf_token: &str) -> Markup {
    if session.undo_in_progress
        || session.finish_in_progress
        || session.shutdown_requested
        || session.archive_publication.locks_undo()
        || session.suspended_current.is_some()
        || !undo_is_safe(&session.current)
    {
        return html! {};
    }
    let Some(completed) = session.completed_reviews.last() else {
        return html! {};
    };
    html! {
        form action="undo" method="post" class="undo" data-undo-form {
            (csrf_field(csrf_token))
            input type="hidden" name="session_id" value=(session.session_id);
            input type="hidden" name="review_id" value=(completed.review_id);
            @if let Some(Current::Ready { frozen, draft, .. }) = &session.current {
                input type="hidden" name="current_token" value=(frozen.token);
                input type="hidden" name="current_instance_id" value=(frozen.instance_id);
                input type="hidden" name="current_draft" value=(draft);
            }
            button class="text-button" type="submit" { "U · Undo previous" }
        }
    }
}

fn generation_error(
    session_id: i64,
    token: u64,
    spec: &DrillSpec,
    message: &str,
    csrf_token: &str,
) -> Markup {
    html! {
        section class="card" {
            div class="deck" { (&spec.deck_name) }
            div class="notice error" role="alert" {
                strong { "Could not prepare this drill." }
                p { (message) }
            }
            div class="actions" {
                form action="regenerate" method="post" {
                    (csrf_field(csrf_token))
                    input type="hidden" name="session_id" value=(session_id);
                    input type="hidden" name="token" value=(token);
                    button class="primary" type="submit" { "Try again" }
                }
                form action="skip" method="post" {
                    (csrf_field(csrf_token))
                    input type="hidden" name="session_id" value=(session_id);
                    input type="hidden" name="token" value=(token);
                    button class="quiet" type="submit" { "Skip" }
                }
            }
        }
    }
}

fn finished_page(session: &SessionState, csrf_token: &str) -> Markup {
    let reviewed = session.completed_reviews.len();
    let skipped = session.completed.saturating_sub(reviewed);
    let retained = session
        .completed_reviews
        .iter()
        .filter(|review| review.accepted_grade != Grade::Forgot)
        .count();
    let retention = format_retention(retained, reviewed);
    let duration = session
        .finished_at
        .map(|finished_at| format_duration(session.started_at, finished_at))
        .unwrap_or_else(|| "—".into());
    let speed = session
        .finished_at
        .map(|finished_at| {
            format_average_duration(session.started_at, finished_at, session.completed)
        })
        .unwrap_or_else(|| "—".into());
    html! {
        main class="finished" {
            section class="card completion-card" {
                h1 { "SESSION COMPLETE" }
                section class="summary-section" aria-labelledby="session-summary" {
                    h2 id="session-summary" { "SESSION" }
                    dl class="stat-grid session-stat-grid" {
                        (stat("RESOLVED", session.completed))
                        (stat("REVIEWED", reviewed))
                        (stat("SKIPPED", skipped))
                        (stat("RETENTION", retention))
                        (stat("DURATION", duration))
                        (stat("AVG / DRILL", speed))
                    }
                }
                @if let Some(history) = &session.completion_history {
                    (collection_summary(history))
                    (history_panel(history))
                } @else {
                    section class="summary-section" aria-labelledby="collection-summary" {
                        h2 id="collection-summary" { "COLLECTION" }
                        p class="muted" { "UNAVAILABLE" }
                        @if let Some(message) = &session.history_error {
                            span class="sr-only" { (message) }
                        }
                    }
                }
            }
            div class="completion-actions" {
                (undo_control(session, csrf_token))
                @if session_transcript_is_ready(session) {
                    a class="text-button" href=(relative_route("/session.md")) download {
                        "Download transcript"
                    }
                }
                form action="shutdown" method="post" data-shutdown-form {
                    (csrf_field(csrf_token))
                    input type="hidden" name="session_id" value=(session.session_id);
                    button class="primary" type="submit" { "Shutdown" }
                }
            }
        }
    }
}

fn stat(label: &str, value: impl ToString) -> Markup {
    html! {
        div {
            dt { (label) }
            dd { (value.to_string()) }
        }
    }
}

fn collection_summary(history: &CompletionHistory) -> Markup {
    let stats = history.stats;
    html! {
        section class="summary-section" aria-labelledby="collection-summary" {
            h2 id="collection-summary" { "COLLECTION" }
            dl class="stat-grid" {
                (stat("DRILLS", stats.total_specs))
                (stat("REVIEWED", stats.total_specs.saturating_sub(stats.unseen_specs)))
                (stat("DUE", stats.due_specs))
                (stat("UNSEEN", stats.unseen_specs))
            }
        }
    }
}

fn history_panel(history: &CompletionHistory) -> Markup {
    html! {
        details class="history" {
            summary { "HISTORY" }
            (review_heatmap(history))
            div class="history-graphs" {
                (bar_chart(
                    "SCHEDULED GRADES · 30 DAYS",
                    &[
                        ("FORGOT", history.recent_grades.forgot),
                        ("HARD", history.recent_grades.hard),
                        ("GOOD", history.recent_grades.good),
                        ("EASY", history.recent_grades.easy),
                    ],
                    "NO SCHEDULED GRADES IN THIS WINDOW",
                ))
                (bar_chart(
                    "SCHEDULING HORIZONS",
                    &[
                        ("<7 DAYS", history.horizons.under_7_days),
                        ("7–29 DAYS", history.horizons.days_7_to_29),
                        ("30–89 DAYS", history.horizons.days_30_to_89),
                        ("90+ DAYS", history.horizons.days_90_plus),
                    ],
                    "NO REVIEWED DRILLS YET",
                ))
            }
        }
    }
}

fn review_heatmap(history: &CompletionHistory) -> Markup {
    let activity: HashMap<Date, usize> = history
        .activity
        .iter()
        .map(|day| (day.date, day.reviews))
        .collect();
    let total: usize = history.activity.iter().map(|day| day.reviews).sum();
    let active_days = history
        .activity
        .iter()
        .filter(|day| day.reviews > 0)
        .count();
    let busiest = history
        .activity
        .iter()
        .map(|day| day.reviews)
        .max()
        .unwrap_or(0);
    html! {
        figure class="heatmap-figure" {
            figcaption { "REVIEWS · 53 WEEKS" }
            div class="heatmap-scroll" {
                div class="heatmap-layout" {
                    div class="heatmap-weekdays" aria-hidden="true" {
                        span { "S" } span {} span { "T" } span {} span { "T" } span {} span { "S" }
                    }
                    div class="heatmap" aria-hidden="true" {
                        @for offset in 0..(53 * 7) {
                            @let date = Date::new(
                                history.activity_start.into_inner() + Duration::days(offset),
                            );
                            @if date > history.today {
                                span class="heat-cell future" {}
                            } @else {
                                @let reviews = activity.get(&date).copied().unwrap_or(0);
                                @let noun = if reviews == 1 { "review" } else { "reviews" };
                                span class=(format!("heat-cell level-{}", heat_level(reviews)))
                                    title=(format!("{date} · {reviews} {noun}")) {}
                            }
                        }
                    }
                }
            }
            p class="sr-only" {
                (total) " reviews across " (active_days) " active days. Busiest day: " (busiest) " reviews."
            }
            @if total == 0 {
                p class="empty-chart" { "NO SCHEDULED REVIEWS IN THIS WINDOW" }
            }
        }
    }
}

fn heat_level(reviews: usize) -> usize {
    match reviews {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        _ => 4,
    }
}

fn bar_chart(title: &str, rows: &[(&str, usize)], empty: &str) -> Markup {
    let maximum = rows.iter().map(|(_, count)| *count).max().unwrap_or(0);
    html! {
        figure class="bar-chart" {
            figcaption { (title) }
            @if maximum == 0 {
                p class="empty-chart" { (empty) }
            } @else {
                @for (label, count) in rows {
                    div class="bar-row" {
                        span class="bar-label" { (label) }
                        span class="bar-track" aria-hidden="true" {
                            span class="bar-fill" style=(format!(
                                "width: {}%",
                                count.saturating_mul(100) / maximum,
                            )) {}
                        }
                        strong { (count) }
                    }
                }
            }
        }
    }
}

fn format_duration(started_at: Timestamp, finished_at: Timestamp) -> String {
    format_seconds(elapsed_seconds(started_at, finished_at))
}

fn format_average_duration(
    started_at: Timestamp,
    finished_at: Timestamp,
    resolved: usize,
) -> String {
    if resolved == 0 {
        return "—".into();
    }
    let elapsed = elapsed_seconds(started_at, finished_at);
    let resolved = i64::try_from(resolved).unwrap_or(i64::MAX);
    format_seconds((elapsed + resolved / 2) / resolved)
}

fn elapsed_seconds(started_at: Timestamp, finished_at: Timestamp) -> i64 {
    (finished_at.into_inner() - started_at.into_inner())
        .num_seconds()
        .max(0)
}

fn format_seconds(seconds: i64) -> String {
    let hours = seconds / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn format_retention(retained: usize, reviewed: usize) -> String {
    if reviewed == 0 {
        return "—".into();
    }
    let percent = (retained.saturating_mul(100) + reviewed / 2) / reviewed;
    format!("{percent}%")
}

fn shutdown_page() -> Markup {
    html! {
        main class="finished" {
            section class="card shutdown-card" {
                h1 { "SHUTTING DOWN" }
                p { "SESSION SAVED" }
            }
        }
    }
}

fn finalization_page(session: &SessionState, csrf_token: &str) -> Markup {
    html! {
        main class="finished" {
            section class="card waiting" data-poll-current[session.finish_in_progress] {
                @if let Some(message) = &session.finish_error {
                    h1 { "FINALIZATION ERROR" }
                    div class="notice error" role="alert" { (message) }
                    form action="retry-finish" method="post" class="actions" {
                        (csrf_field(csrf_token))
                        input type="hidden" name="session_id" value=(session.session_id);
                        button class="primary" type="submit" { "Retry" }
                    }
                } @else {
                    h1 { "FINALIZING" }
                    p class="status" { "RECORDING SESSION" }
                }
            }
        }
    }
}

fn csrf_field(csrf_token: &str) -> Markup {
    html! {
        input type="hidden" name="csrf_token" value=(csrf_token);
    }
}

fn verdict_class(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Pass => "pass",
        Verdict::Partial => "partial",
        Verdict::Fail => "fail",
        Verdict::Uncertain | Verdict::Invalid => "uncertain",
    }
}

fn reject(message: &str) -> Response {
    error_page(StatusCode::BAD_REQUEST, message)
}

fn stale_submission() -> Response {
    error_page(
        StatusCode::CONFLICT,
        "That form belongs to an older drill state and was not applied.",
    )
}

fn error_page(status: StatusCode, message: &str) -> Response {
    let body = html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { "Hashdrills error" }
                style { (PreEscaped(STYLES)) }
            }
            body {
                main {
                    section class="card" {
                        h1 { "Hashdrills" }
                        div class="notice error" role="alert" { (message) }
                        p { a href="." { "Return to the current drill" } }
                    }
                }
            }
        }
    };
    (status, Html(body.into_string())).into_response()
}

const STYLES: &str = r#"
:root {
  color-scheme: light;
  font-family: Verdana, Geneva, sans-serif;
  --ink: #171716;
  --ui-ink: #575752;
  --line: #aaa9a1;
  --line-strong: #85857e;
  --soft: #deded8;
  background: #f2f2ef;
  color: var(--ink);
}
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; }
main { width: min(800px, calc(100% - 2rem)); margin: 2rem auto; }
body, button, textarea { font-variant-numeric: tabular-nums; }
header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: .5rem; }
.brand { margin: 0; color: var(--ui-ink); font-size: 1rem; font-weight: 700; text-transform: uppercase; }
.progress-label, .deck, .muted, .shortcut { color: #5d5d59; }
.progress { height: 4px; border: 1px solid var(--line); margin-bottom: 1rem; }
.progress-fill { height: 100%; background: var(--ui-ink); }
.card { background: #fff; border: 1px solid var(--line-strong); padding: clamp(1rem, 4vw, 2rem); }
.deck-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: .75rem 1.25rem; margin-bottom: 1.6rem; }
.deck-header .deck { margin-bottom: 0; }
.deck { display: inline-block; margin-bottom: 1.6rem; border-left: 3px solid var(--ui-ink); padding: .4rem .55rem; background: #e9e9e4; color: var(--ink); font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.source { flex: none; color: var(--ui-ink); font-size: .72rem; font-weight: 600; text-decoration: none; }
.source:hover { text-decoration: underline; }
.question, .response, .reference { overflow-wrap: anywhere; line-height: 1.55; }
.response { white-space: pre-wrap; }
.question { font-size: clamp(1.55rem, 4vw, 2.25rem); font-weight: 700; line-height: 1.35; margin-bottom: 2rem; }
.question.question-long { font-size: clamp(1.35rem, 3.2vw, 1.9rem); }
.question.question-very-long { font-size: clamp(1.15rem, 2.6vw, 1.55rem); line-height: 1.45; }
.question.compact { font-size: clamp(1.2rem, 2.5vw, 1.5rem); line-height: 1.4; padding-bottom: 1.25rem; border-bottom: 1px solid var(--line); }
.result-question { margin-bottom: .8rem; padding-bottom: 0; border-bottom: 0; }
.goal { margin: 0 0 1.4rem; text-align: left; }
.goal > summary { padding: .55rem 0; cursor: pointer; color: var(--ui-ink); font-size: .7rem; font-weight: 700; letter-spacing: .08em; }
.goal > summary:hover { text-decoration: underline; }
.goal .reference { margin-top: .6rem; }
.rich-text { white-space: normal; }
.rich-text > :first-child { margin-top: 0; }
.rich-text > :last-child { margin-bottom: 0; }
.rich-text p, .rich-text ul, .rich-text ol, .rich-text blockquote, .rich-text pre, .rich-text table { margin: 0 0 1em; }
.rich-text ul, .rich-text ol { padding-left: 1.5em; }
.rich-text li + li { margin-top: .3em; }
.rich-text blockquote { border-left: 3px solid var(--line); padding-left: .8em; color: var(--ui-ink); }
.rich-text img { display: block; max-width: 100%; max-height: 70vh; width: auto; height: auto; object-fit: contain; }
.rich-text audio, .rich-text video { display: block; max-width: 100%; margin: 0 0 1em; }
.rich-text table { display: block; width: max-content; max-width: 100%; overflow-x: auto; border-collapse: collapse; font-size: 1rem; font-weight: 400; line-height: 1.4; }
.rich-text th, .rich-text td { border: 1px solid var(--line); padding: .45rem .6rem; text-align: left; white-space: nowrap; }
.rich-text th { background: #ecece7; font-weight: 700; }
.rich-text pre { max-width: 100%; overflow-x: auto; border: 1px solid var(--line); padding: .8rem; background: #f3f3ef; white-space: pre; font-size: .82rem; font-weight: 400; line-height: 1.45; }
.rich-text code { font-family: Menlo, Monaco, Consolas, monospace; font-size: .82em; font-weight: 400; }
.rich-text :not(pre) > code { border: 1px solid var(--line); padding: .08em .25em; background: #f3f3ef; }
.media-fallback { display: inline-block; border: 1px dashed var(--line); padding: .35rem .5rem; color: var(--ui-ink); font-size: .72rem; font-weight: 400; }
.math-display { display: block; max-width: 100%; overflow-x: auto; padding: .15em 0; }
h1 { margin: .25rem 0 1rem; }
h2 { color: var(--ui-ink); font-size: .75rem; margin: 1.5rem 0 .5rem; letter-spacing: .08em; }
label { display: block; color: var(--ui-ink); font-size: .9rem; font-weight: 600; margin-bottom: .45rem; }
textarea { width: 100%; min-height: 12rem; resize: vertical; border: 1px solid var(--line); padding: .8rem; font: inherit; line-height: 1.5; background: #fff; color: inherit; }
:focus-visible { outline: 2px solid #707069; outline-offset: 2px; }
[tabindex="-1"]:focus-visible { outline: none; }
button { border: 1px solid var(--line); border-radius: 0; padding: .7rem .9rem; font: inherit; font-weight: 600; cursor: pointer; background: #fff; color: var(--ui-ink); }
button:hover { text-decoration: underline; }
.primary { border-color: var(--line-strong); color: #353532; background: var(--soft); }
.quiet { color: var(--ui-ink); background: #fff; }
.text-button { border: 0; padding: .25rem 0; background: transparent; color: var(--ui-ink); text-decoration: underline; }
.actions { display: flex; gap: .65rem; justify-content: flex-end; align-items: center; margin-top: 1rem; }
.result-actions { margin-top: 2rem; }
.split { justify-content: space-between; }
.split > form { display: flex; align-items: center; gap: .6rem; }
.check { display: flex; justify-content: space-between; gap: 1rem; margin-top: 1.25rem; border: 1px solid var(--line); padding: .75rem; }
.check.fail, .check.partial, .check.error { color: #fff; background: #50504c; }
.reference, .response, .datum { padding: .75rem; border: 1px solid var(--line); background: #f7f7f4; }
.notice { border: 1px solid var(--line); padding: .75rem; margin: 1rem 0; line-height: 1.45; }
.notice.error { border-left-width: 5px; }
.notice.neutral { border-style: dashed; }
.global-notice { max-width: 88rem; margin: 1rem auto; }
.rating-sheet { border-top: 1px solid var(--line); margin-top: 1.5rem; padding-top: 1rem; }
.grades { display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; }
.grades button { text-align: left; }
.edit-answer { margin-top: 1.75rem; }
kbd { display: inline-block; min-width: 1.5rem; border: 1px solid var(--line); padding: .05rem .25rem; text-align: center; }
.shortcut, .status { font-size: .78rem; line-height: 1.45; }
.status { min-height: 1.2em; font-weight: 800; }
.result-grid { display: grid; grid-template-columns: 3fr 1fr; gap: 1rem; }
.result-grid > div { display: flex; flex-direction: column; }
.result-grid .response, .result-grid .datum { flex: 1; }
.undo { margin-top: .7rem; text-align: right; }
.waiting { text-align: left; }
.finished { text-align: center; }
.completion-card { text-align: left; }
.completion-card > h1 { margin: .25rem 0 2rem; text-align: center; font-size: clamp(2rem, 6vw, 3rem); }
.summary-section + .summary-section { margin-top: 2rem; }
.summary-section h2 { margin-top: 0; }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; margin: 0; background: var(--line); border: 1px solid var(--line); }
.session-stat-grid { grid-template-columns: repeat(6, 1fr); }
.stat-grid > div { min-width: 0; padding: .8rem; background: #fff; }
.stat-grid dt { color: #5d5d59; font-size: .67rem; font-weight: 700; letter-spacing: .06em; }
.stat-grid dd { overflow-wrap: anywhere; margin: .35rem 0 0; font-size: clamp(1rem, 3vw, 1.35rem); font-weight: 700; }
.history { margin-top: 2rem; border-top: 1px solid var(--line); }
.history > summary { padding: 1rem 0; cursor: pointer; font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-align: left; }
.history > summary:hover { text-decoration: underline; }
.heatmap-figure, .bar-chart { margin: 1rem 0 0; text-align: left; }
.heatmap-figure figcaption, .bar-chart figcaption { margin-bottom: .75rem; font-size: .68rem; font-weight: 700; letter-spacing: .06em; }
.heatmap-scroll { overflow-x: auto; padding-bottom: .35rem; }
.heatmap-layout { display: grid; grid-template-columns: .8rem max-content; gap: .4rem; width: max-content; }
.heatmap-weekdays { display: grid; grid-template-rows: repeat(7, 10px); gap: 2px; color: #5d5d59; font-size: .5rem; line-height: 10px; }
.heatmap { display: grid; grid-template-rows: repeat(7, 10px); grid-auto-flow: column; grid-auto-columns: 10px; gap: 2px; }
.heat-cell { width: 10px; height: 10px; border: 1px solid #ebedf0; background: #ebedf0; cursor: help; }
.heat-cell.level-1 { border-color: #9be9a8; background: #9be9a8; }
.heat-cell.level-2 { border-color: #40c463; background: #40c463; }
.heat-cell.level-3 { border-color: #30a14e; background: #30a14e; }
.heat-cell.level-4 { border-color: #216e39; background: #216e39; }
.heat-cell.future { visibility: hidden; }
.history-graphs { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem; }
.bar-row { display: grid; grid-template-columns: 6.8rem 1fr 2rem; gap: .6rem; align-items: center; margin-top: .65rem; font-size: .68rem; }
.bar-label { color: #5d5d59; }
.bar-track { height: .65rem; border: 1px solid var(--line); background: #fff; }
.bar-fill { display: block; height: 100%; background: var(--ui-ink); }
.empty-chart { color: #5d5d59; font-size: .68rem; line-height: 1.5; }
.completion-actions { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-top: 1.25rem; }
.completion-actions .undo { margin: 0; }
.shutdown-card p { margin-bottom: 0; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
@media (max-width: 560px) {
  main { margin: 1rem auto; }
  .grades { grid-template-columns: repeat(2, 1fr); }
  .result-grid { grid-template-columns: 1fr; }
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
  .history-graphs { grid-template-columns: 1fr; gap: 1rem; }
  .completion-actions { align-items: stretch; flex-direction: column-reverse; }
  .completion-actions .undo { text-align: center; }
  .actions, .split { align-items: stretch; flex-direction: column; }
  button { min-height: 3rem; }
}
@media (min-width: 561px) and (max-width: 760px) {
  .session-stat-grid { grid-template-columns: repeat(3, 1fr); }
}
"#;

const SCRIPT: &str = r#"
(() => {
  document.addEventListener('DOMContentLoaded', async () => {
    let authoredMacros = {};
    try {
      const response = await fetch('assets/macros.json', {
        cache: 'no-store',
        credentials: 'same-origin',
        headers: { Accept: 'application/json' },
      });
      if (response.ok) {
        const candidate = await response.json();
        if (candidate !== null && typeof candidate === 'object' && !Array.isArray(candidate)) {
          authoredMacros = candidate;
        }
      }
    } catch (_) {
      // Math still renders with KaTeX's built-in commands when no custom map
      // is available.
    }
    if (window.katex) {
      for (const element of document.querySelectorAll('.math-inline, .math-display')) {
        window.katex.render(element.textContent, element, {
          displayMode: element.classList.contains('math-display'),
          throwOnError: false,
          trust: false,
          maxSize: 20,
          maxExpand: 1000,
          macros: { ...authoredMacros },
        });
      }
    }
    if (window.hljs) {
      for (const block of document.querySelectorAll('.rich-text pre code')) {
        window.hljs.highlightElement(block);
      }
    }
  });

  const isTextEntry = (target) => target instanceof Element &&
    Boolean(target.closest('textarea,input,select,[contenteditable="true"],[contenteditable=""],[role="textbox"]'));
  const isAction = (target) => target instanceof Element &&
    Boolean(target.closest('button,a,input,textarea,select,[contenteditable],[role="button"]'));
  const composing = (event) => event.isComposing || event.keyCode === 229;

  const answer = document.querySelector('[data-answer-form] textarea');
  if (answer) {
    answer.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !composing(event)) {
        event.preventDefault();
        answer.form.requestSubmit();
      }
    });
  }

  const rateForm = document.querySelector('[data-rate-form]');
  const regradeForm = document.querySelector('[data-regrade-form]');
  for (const form of [rateForm, regradeForm]) {
    if (form) form.addEventListener('submit', (event) => {
      if (form.dataset.submitting) {
        event.preventDefault();
        return;
      }
      form.dataset.submitting = 'true';
    });
  }

  const defaultForm = document.querySelector('[data-default-form]');
  const decisionForms = document.querySelectorAll('[data-decision-form]');
  let decisionSubmitted = false;
  const submitDefault = () => {
    if (!defaultForm || decisionSubmitted) return;
    defaultForm.requestSubmit();
  };
  for (const form of decisionForms) {
    form.addEventListener('submit', (event) => {
      if (decisionSubmitted) {
        event.preventDefault();
        return;
      }
      decisionSubmitted = true;
    });
  }

  const result = document.querySelector('[data-result-focus]');
  if (result) window.requestAnimationFrame(() => result.focus({preventScroll: true}));
  const rating = document.querySelector('[data-rating-focus]');
  if (rating) window.requestAnimationFrame(() => rating.focus());
  const undoForm = document.querySelector('[data-undo-form]');
  if (undoForm) undoForm.addEventListener('submit', (event) => {
    if (undoForm.dataset.submitting) {
      event.preventDefault();
      return;
    }
    undoForm.dataset.submitting = 'true';
    const draft = undoForm.querySelector('[name="current_draft"]');
    const textarea = document.querySelector('[data-answer-form] textarea');
    if (draft && textarea) draft.value = textarea.value;
    for (const button of undoForm.querySelectorAll('button')) button.disabled = true;
  });
  const shutdownForm = document.querySelector('[data-shutdown-form]');
  if (shutdownForm) shutdownForm.addEventListener('submit', () => {
    for (const button of shutdownForm.querySelectorAll('button')) button.disabled = true;
  });
  if (document.querySelector('[data-poll-current]')) {
    const poll = async () => {
      try {
        const response = await fetch('.', {headers: {'Accept': 'text/html'}});
        const body = await response.text();
        if (!body.includes('data-poll-current')) {
          window.location.reload();
          return;
        }
      } catch (_) {}
      window.setTimeout(poll, 500);
    };
    window.setTimeout(poll, 500);
  }

  document.addEventListener('keydown', (event) => {
    if (event.defaultPrevented || event.repeat || event.metaKey || event.ctrlKey || event.altKey || composing(event)) return;
    if (isTextEntry(event.target)) return;
    const key = event.key.toLowerCase();
    const gradeForm = rateForm || regradeForm;
    if (gradeForm && ['1', '2', '3', '4'].includes(key)) {
      event.preventDefault();
      const grades = ['forgot', 'hard', 'good', 'easy'];
      gradeForm.querySelector(`button[value="${grades[Number(key) - 1]}"]`)?.click();
      return;
    }
    if (key === 'u') {
      const form = document.querySelector('[data-back-form], [data-undo-form]');
      if (form) {
        event.preventDefault();
        form.requestSubmit();
      }
      return;
    }
    if (event.key === 'Enter' && !event.shiftKey && defaultForm && !isAction(event.target)) {
      event.preventDefault();
      submitDefault();
    }
  });
})();
"#;

#[cfg(test)]
mod tests {
    use std::fs::write;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use crate::error::fail;
    use crate::model::ModelFuture;
    use crate::spec::parse_path;

    use super::*;

    struct FakeBackend {
        verdict: Verdict,
        question: String,
        generations: AtomicUsize,
        evaluations: AtomicUsize,
    }

    impl FakeBackend {
        fn new(verdict: Verdict) -> Self {
            Self {
                verdict,
                question: "QUESTION-SENTINEL".into(),
                generations: AtomicUsize::new(0),
                evaluations: AtomicUsize::new(0),
            }
        }

        fn with_question(mut self, question: &str) -> Self {
            self.question = question.into();
            self
        }
    }

    impl ModelBackend for FakeBackend {
        fn generate<'a>(&'a self, _spec: &'a DrillSpec) -> ModelFuture<'a, GeneratedInstance> {
            self.generations.fetch_add(1, Ordering::SeqCst);
            let question = self.question.clone();
            Box::pin(async move {
                Ok(GeneratedInstance {
                    question,
                    target: "TARGET-SENTINEL".into(),
                    rubric: "RUBRIC-SENTINEL".into(),
                    model: "fake".into(),
                    protocol_version: crate::model::PROTOCOL_VERSION,
                })
            })
        }

        fn evaluate<'a>(
            &'a self,
            _instance: &'a GeneratedInstance,
            _response: &'a str,
        ) -> ModelFuture<'a, Evaluation> {
            self.evaluations.fetch_add(1, Ordering::SeqCst);
            let verdict = self.verdict;
            Box::pin(async move {
                Ok(Evaluation {
                    verdict,
                    feedback: "FEEDBACK-SENTINEL".into(),
                    model: "fake".into(),
                    protocol_version: crate::model::PROTOCOL_VERSION,
                })
            })
        }
    }

    #[derive(Default)]
    struct FakeRecorder {
        freezes: AtomicUsize,
        ungraded_attempts: AtomicUsize,
        evaluations: AtomicUsize,
        graded_attempts: AtomicUsize,
        undos: AtomicUsize,
        finishes: AtomicUsize,
        history_queries: AtomicUsize,
        fail_finishes: AtomicBool,
        grades: Mutex<Vec<(Grade, ReviewResolution)>>,
        learner_grades: Mutex<Vec<Grade>>,
    }

    impl SessionRecorder for FakeRecorder {
        fn freeze_instance(
            &self,
            _session_id: i64,
            _spec: &DrillSpec,
            _instance: &GeneratedInstance,
            _at: Timestamp,
        ) -> Fallible<i64> {
            Ok(self.freezes.fetch_add(1, Ordering::SeqCst) as i64 + 1)
        }

        fn record_attempt(
            &self,
            _instance_id: i64,
            _response: &str,
            learner_grade: Grade,
            _at: Timestamp,
        ) -> Fallible<i64> {
            self.learner_grades.lock().unwrap().push(learner_grade);
            Ok(self.ungraded_attempts.fetch_add(1, Ordering::SeqCst) as i64 + 1)
        }

        fn attach_evaluation(
            &self,
            _attempt_id: i64,
            _evaluation: &Evaluation,
            _at: Timestamp,
        ) -> Fallible<()> {
            self.evaluations.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn accept_grade(
            &self,
            _attempt_id: i64,
            grade: Grade,
            resolution: ReviewResolution,
            _at: Timestamp,
        ) -> Fallible<i64> {
            self.grades.lock().unwrap().push((grade, resolution));
            Ok(self.graded_attempts.fetch_add(1, Ordering::SeqCst) as i64 + 1)
        }

        fn undo_review(&self, _review_id: i64, _at: Timestamp) -> Fallible<()> {
            self.undos.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn finish_session(&self, _session_id: i64, _at: Timestamp) -> Fallible<()> {
            self.finishes.fetch_add(1, Ordering::SeqCst);
            if self.fail_finishes.load(Ordering::SeqCst) {
                return fail("FINISH-SENTINEL");
            }
            Ok(())
        }

        fn completion_history_for_specs(
            &self,
            today: Date,
            specs: &[DrillSpec],
        ) -> Fallible<CompletionHistory> {
            self.history_queries.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionHistory {
                stats: crate::storage::StorageStats {
                    total_specs: specs.len(),
                    unseen_specs: specs.len(),
                    due_specs: 0,
                    frozen_instances: 0,
                    attempts: 0,
                    reviews: 0,
                    completed_sessions: 0,
                },
                today,
                activity_start: today,
                activity: Vec::new(),
                recent_grades: crate::storage::GradeCounts::default(),
                horizons: crate::storage::SchedulingHorizons::default(),
            })
        }
    }

    fn spec() -> DrillSpec {
        parsed_spec("Authored question")
    }

    fn parsed_spec(question: &str) -> DrillSpec {
        let directory = tempfile::tempdir().unwrap();
        write(
            directory.path().join("Practice.md"),
            format!("G: GOAL-SENTINEL\nQ: {question}\nA: Authored target\n"),
        )
        .unwrap();
        parse_path(directory.path()).unwrap().remove(0)
    }

    fn test_timestamp(value: &str) -> Timestamp {
        Timestamp::try_from(value.to_string()).unwrap()
    }

    fn completed_review(
        spec: DrillSpec,
        model: &str,
        identifier: i64,
        generated_at: Timestamp,
        reviewed_at: Timestamp,
    ) -> CompletedReview {
        CompletedReview {
            result: ReviewResult {
                rated: RatedAnswer {
                    submitted: SubmittedAnswer {
                        frozen: FrozenQuestion {
                            token: identifier as u64,
                            spec,
                            instance_id: identifier,
                            generated_at,
                            instance: GeneratedInstance {
                                question: format!("Generated question {identifier}"),
                                target: format!("Generated target {identifier}"),
                                rubric: format!("Generated rubric {identifier}"),
                                model: model.into(),
                                protocol_version: crate::model::PROTOCOL_VERSION,
                            },
                        },
                        response: format!("Learner response {identifier}"),
                        notice: None,
                    },
                    learner_grade: Grade::Good,
                    attempt_id: identifier,
                },
                check: CheckResult::Evaluation(Evaluation {
                    verdict: Verdict::Pass,
                    feedback: "OK".into(),
                    model: "judge".into(),
                    protocol_version: crate::model::PROTOCOL_VERSION,
                }),
                default_grade: Some(Grade::Good),
                notice: None,
                regrading: false,
            },
            review_id: identifier,
            reviewed_at,
            accepted_grade: Grade::Good,
            resolution: ReviewResolution::AiConfirmed,
        }
    }

    fn completed_test_state(
        completed_reviews: Vec<CompletedReview>,
        archive_dir: Option<PathBuf>,
    ) -> (
        AppState,
        Arc<FakeRecorder>,
        tokio::sync::oneshot::Receiver<()>,
    ) {
        let specs = completed_reviews
            .iter()
            .map(|completed| completed.result.rated.submitted.frozen.spec.clone())
            .collect::<Vec<_>>();
        let recorder = Arc::new(FakeRecorder::default());
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let state = AppState::new(
            AccessControl::disabled_for_tests(),
            1,
            Vec::new(),
            specs,
            PathBuf::from("/"),
            archive_dir,
            Arc::new(FakeBackend::new(Verdict::Pass)),
            recorder.clone(),
            test_timestamp("2026-07-31T12:00:00.000"),
            Some(shutdown_tx),
        );
        {
            let mut mutable = state.mutable.lock().unwrap();
            mutable.total = completed_reviews.len();
            mutable.completed = completed_reviews.len();
            mutable.transcript_entries = completed_reviews
                .iter()
                .cloned()
                .map(SessionTranscriptEntry::Reviewed)
                .collect();
            mutable.completed_reviews = completed_reviews;
            mutable.finish_recorded = true;
            mutable.finished_at = Some(test_timestamp("2026-07-31T12:10:00.000"));
        }
        (state, recorder, shutdown_rx)
    }

    async fn serve_test_app(
        verdict: Verdict,
    ) -> (
        String,
        Arc<FakeBackend>,
        Arc<FakeRecorder>,
        tokio::task::JoinHandle<()>,
    ) {
        serve_test_app_with_specs(verdict, vec![spec()]).await
    }

    async fn serve_test_app_with_specs(
        verdict: Verdict,
        specs: Vec<DrillSpec>,
    ) -> (
        String,
        Arc<FakeBackend>,
        Arc<FakeRecorder>,
        tokio::task::JoinHandle<()>,
    ) {
        let backend = Arc::new(FakeBackend::new(verdict));
        let recorder = Arc::new(FakeRecorder::default());
        let queue = specs.clone();
        let collection_root = specs
            .first()
            .and_then(|spec| spec.path.parent())
            .unwrap_or_else(|| Path::new("/"))
            .to_path_buf();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let state = AppState::new(
            AccessControl::disabled_for_tests(),
            1,
            queue,
            specs,
            collection_root,
            None,
            backend.clone(),
            recorder.clone(),
            Timestamp::now(),
            Some(shutdown_tx),
        );
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router(state))
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        (format!("http://{address}"), backend, recorder, task)
    }

    struct SecurityTestApp {
        origin: String,
        root_url: String,
        launch_url: String,
        base_path: String,
        csrf_token: String,
        backend: Arc<FakeBackend>,
        recorder: Arc<FakeRecorder>,
        task: tokio::task::JoinHandle<()>,
        _directory: tempfile::TempDir,
    }

    async fn serve_security_test_app(auth_enabled: bool) -> SecurityTestApp {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let origin = format!("http://{address}");
        let launch = LaunchAccess::new(auth_enabled, "127.0.0.1", address, None).unwrap();
        let base_path = launch.control.base_path().to_string();
        let csrf_token = launch.control.csrf_token().to_string();
        let launch_url = launch.browser_url.clone();

        let directory = tempfile::tempdir().unwrap();
        write(
            directory.path().join("Practice.md"),
            "G: GOAL-SENTINEL\nQ: Authored question\nA: Authored target\n",
        )
        .unwrap();
        write(directory.path().join("image.png"), b"PNG-SENTINEL").unwrap();
        let specs = parse_path(directory.path()).unwrap();
        let backend = Arc::new(
            FakeBackend::new(Verdict::Pass)
                .with_question("QUESTION-SENTINEL\n\n![test image](image.png)"),
        );
        let recorder = Arc::new(FakeRecorder::default());
        let state = AppState::new(
            launch.control,
            1,
            specs.clone(),
            specs,
            directory.path().to_path_buf(),
            None,
            backend.clone(),
            recorder.clone(),
            Timestamp::now(),
            None,
        );
        let task = tokio::spawn(async move {
            axum::serve(listener, router(state)).await.unwrap();
        });

        SecurityTestApp {
            root_url: format!("{origin}{base_path}/"),
            origin,
            launch_url,
            base_path,
            csrf_token,
            backend,
            recorder,
            task,
            _directory: directory,
        }
    }

    fn redirectless_client() -> reqwest::Client {
        reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap()
    }

    async fn submit_answer(client: &reqwest::Client, base: &str) -> String {
        client
            .post(format!("{base}/answer"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("response", "RESPONSE-SENTINEL"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    async fn rate(client: &reqwest::Client, base: &str, grade: &str) -> String {
        client
            .post(format!("{base}/rate"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("grade", grade),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn answer_is_hidden_until_rating_then_pass_waits_for_acceptance() {
        let (base, backend, recorder, task) = serve_test_app(Verdict::Pass).await;
        let client = reqwest::Client::new();

        let question = client
            .get(format!("{base}/"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(question.contains("QUESTION-SENTINEL"));
        assert!(question.contains("placeholder=\"Answer\""));
        assert!(question.contains("class=\"sr-only\" for=\"response\">Answer</label>"));
        assert!(!question.contains(">ANSWER</label>"));
        assert!(!question.contains("TARGET-SENTINEL"));
        assert!(!question.contains("RUBRIC-SENTINEL"));
        assert!(!question.contains("GOAL-SENTINEL"));
        assert!(!question.contains("ENTER submit"));
        assert!(question.contains("SHIFT+ENTER newline"));
        assert!(question.contains("<kbd aria-label=\"Enter\">↵</kbd>"));
        assert_eq!(recorder.freezes.load(Ordering::SeqCst), 1);

        let old_session = client
            .post(format!("{base}/answer"))
            .form(&[
                ("session_id", "999"),
                ("token", "1"),
                ("instance_id", "1"),
                ("response", "stale session"),
            ])
            .send()
            .await
            .unwrap();
        assert_eq!(old_session.status(), StatusCode::CONFLICT);

        let rating = submit_answer(&client, &base).await;
        assert!(!rating.contains("SELF-RATE"));
        assert!(!rating.contains("Forgot skips the AI check"));
        assert!(!rating.contains("Claimed success is checked"));
        assert!(!rating.contains("1–4 select"));
        assert!(rating.contains("aria-label=\"Rate answer\""));
        assert!(rating.contains("<kbd>U</kbd> Edit answer"));
        assert!(rating.contains("RESPONSE-SENTINEL"));
        assert!(!rating.contains("<h2>ANSWER</h2>"));
        assert!(!rating.contains("TARGET-SENTINEL"));
        assert!(!rating.contains("RUBRIC-SENTINEL"));
        assert!(!rating.contains("GOAL-SENTINEL"));
        assert_eq!(backend.evaluations.load(Ordering::SeqCst), 0);
        assert_eq!(recorder.ungraded_attempts.load(Ordering::SeqCst), 0);

        let result = rate(&client, &base, "good").await;
        assert!(result.contains("YEA"));
        assert!(result.contains(">✓</span>"));
        assert!(!result.contains(">Pass<"));
        assert!(result.contains("FEEDBACK-SENTINEL"));
        let check = result
            .split_once("<div class=\"check pass\">")
            .unwrap()
            .1
            .split_once("</div>")
            .unwrap()
            .0;
        assert!(check.contains("FEEDBACK-SENTINEL"));
        assert!(!result.contains("EVIDENCE-SENTINEL"));
        assert!(result.contains("TARGET-SENTINEL"));
        assert!(!result.contains("<h2>ANSWER</h2>"));
        assert!(!result.contains("<h2>SELF-RATING</h2>"));
        assert!(result.contains("<details class=\"goal\"><summary>GOAL</summary>"));
        assert!(result.contains("GOAL-SENTINEL"));
        assert!(!result.contains("<details class=\"goal\" open>"));
        assert!(
            result.find("result-question").unwrap()
                < result.find("<details class=\"goal\">").unwrap()
        );
        assert!(
            result.find("<details class=\"goal\">").unwrap()
                < result.find("<div class=\"result-grid\">").unwrap()
        );
        assert!(result.contains("<h2>CRITERIA</h2>"));
        assert!(!result.contains("TARGET / CRITERIA"));
        assert!(result.contains(">Continue</button>"));
        assert!(!result.contains("data-auto-advance"));
        assert!(!result.contains("data-countdown"));
        assert!(!result.contains("data-pause-timer"));
        assert!(!result.contains("DEFAULT"));
        assert!(!result.contains("ENTER continue"));
        assert!(!result.contains("U available after save"));
        assert!(!result.contains("Keep my rating"));
        assert_eq!(backend.evaluations.load(Ordering::SeqCst), 1);
        assert_eq!(recorder.ungraded_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(recorder.evaluations.load(Ordering::SeqCst), 1);
        assert_eq!(recorder.graded_attempts.load(Ordering::SeqCst), 0);

        let finished = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("SESSION COMPLETE"));
        assert!(finished.contains("Undo previous"));
        assert!(finished.contains("<h2 id=\"session-summary\">SESSION</h2>"));
        assert!(finished.contains("<h2 id=\"collection-summary\">COLLECTION</h2>"));
        assert!(finished.contains("<summary>HISTORY</summary>"));
        assert!(finished.contains("REVIEWS · 53 WEEKS"));
        assert!(finished.contains("SCHEDULED GRADES · 30 DAYS"));
        assert!(finished.contains("SCHEDULING HORIZONS"));
        assert!(finished.contains(">Shutdown</button>"));
        assert!(finished.contains("href=\"session.md\" download"));
        assert!(finished.contains("Download transcript"));
        assert!(finished.contains("Verdana, Geneva, sans-serif"));
        assert!(finished.contains("<dt>RETENTION</dt><dd>100%</dd>"));
        assert!(finished.contains("<dt>AVG / DRILL</dt>"));
        assert!(finished.contains("title="));
        assert!(finished.contains(" · 0 reviews"));
        assert_eq!(recorder.graded_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(recorder.history_queries.load(Ordering::SeqCst), 1);
        assert_eq!(
            recorder.grades.lock().unwrap()[0],
            (Grade::Good, ReviewResolution::AiConfirmed)
        );

        let undone = client
            .post(format!("{base}/undo"))
            .form(&[("session_id", "1"), ("review_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(undone.contains("REPLACEMENT GRADE"));
        assert!(undone.contains("name=\"token\" value=\"2\""));
        assert!(!undone.contains(">Continue</button>"));
        assert_eq!(recorder.undos.load(Ordering::SeqCst), 1);

        let regraded = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "2"),
                ("instance_id", "1"),
                ("action", "regrade"),
                ("grade", "hard"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(regraded.contains("SESSION COMPLETE"));
        assert!(regraded.contains("<dt>RETENTION</dt><dd>100%</dd>"));
        assert_eq!(
            recorder.grades.lock().unwrap()[1],
            (Grade::Hard, ReviewResolution::UndoRegrade)
        );
        assert_eq!(recorder.finishes.load(Ordering::SeqCst), 2);

        let second_undo = client
            .post(format!("{base}/undo"))
            .form(&[("session_id", "1"), ("review_id", "2")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(second_undo.contains("name=\"token\" value=\"3\""));
        let stale_regrade = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "2"),
                ("instance_id", "1"),
                ("action", "regrade"),
                ("grade", "easy"),
            ])
            .send()
            .await
            .unwrap();
        assert_eq!(stale_regrade.status(), StatusCode::CONFLICT);

        task.abort();
    }

    #[tokio::test]
    async fn forgot_skips_ai_and_records_learner_forgot() {
        let (base, backend, recorder, task) = serve_test_app(Verdict::Pass).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;

        let result = rate(&client, &base, "forgot").await;
        assert!(result.contains("AI check skipped"));
        assert!(result.contains("TARGET-SENTINEL"));
        assert!(result.contains(">Continue</button>"));
        assert!(!result.contains("data-auto-advance"));
        assert_eq!(backend.evaluations.load(Ordering::SeqCst), 0);
        assert_eq!(recorder.evaluations.load(Ordering::SeqCst), 0);

        let finished = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("Download transcript"));
        assert_eq!(
            recorder.grades.lock().unwrap()[0],
            (Grade::Forgot, ReviewResolution::LearnerForgot)
        );

        let transcript = client
            .get(format!("{base}/session.md"))
            .send()
            .await
            .unwrap();
        assert_eq!(transcript.status(), StatusCode::OK);
        assert_eq!(
            transcript.headers()[CONTENT_TYPE],
            "text/markdown; charset=utf-8"
        );
        let transcript = transcript.text().await.unwrap();
        assert!(transcript.contains("- Status: reviewed"));
        assert!(transcript.contains("- Self-rating: forgot"));
        assert!(transcript.contains("- Effective rating: forgot"));
        assert!(transcript.contains("- Resolution: learner forgot"));
        assert!(transcript.contains("- AI check: skipped (Forgot self-rating)"));
        assert!(transcript.contains("RESPONSE-SENTINEL"));

        let replacement = client
            .post(format!("{base}/undo"))
            .form(&[("session_id", "1"), ("review_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(!replacement.contains("SELF-RATE"));
        assert!(replacement.contains("aria-label=\"Rate answer\""));
        assert!(replacement.contains("<kbd>U</kbd> Edit answer"));
        assert!(replacement.contains("RESPONSE-SENTINEL"));
        assert!(!replacement.contains("TARGET-SENTINEL"));
        assert!(replacement.contains("value=\"2\""));
        assert_eq!(recorder.freezes.load(Ordering::SeqCst), 2);

        let checked = client
            .post(format!("{base}/rate"))
            .form(&[
                ("session_id", "1"),
                ("token", "2"),
                ("instance_id", "2"),
                ("grade", "good"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(checked.contains("YEA"));
        assert_eq!(backend.evaluations.load(Ordering::SeqCst), 1);
        task.abort();
    }

    #[tokio::test]
    async fn skipping_an_undone_forgot_restores_the_already_frozen_next_drill() {
        let specs = vec![spec(), parsed_spec("Second authored question")];
        let (base, _backend, recorder, task) =
            serve_test_app_with_specs(Verdict::Pass, specs).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;
        rate(&client, &base, "forgot").await;
        let next = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(next.contains("name=\"token\" value=\"2\""));
        assert_eq!(recorder.freezes.load(Ordering::SeqCst), 2);

        let replacement = client
            .post(format!("{base}/undo"))
            .form(&[
                ("session_id", "1"),
                ("review_id", "1"),
                ("current_token", "2"),
                ("current_instance_id", "2"),
                ("current_draft", "NEXT-DRAFT"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(replacement.contains("name=\"token\" value=\"3\""));
        assert_eq!(recorder.freezes.load(Ordering::SeqCst), 3);

        let editable = client
            .post(format!("{base}/back"))
            .form(&[("session_id", "1"), ("token", "3"), ("instance_id", "3")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(editable.contains("name=\"token\" value=\"4\""));
        let restored = client
            .post(format!("{base}/skip"))
            .form(&[("session_id", "1"), ("token", "4"), ("instance_id", "3")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(restored.contains("name=\"token\" value=\"2\""));
        assert!(restored.contains("name=\"instance_id\" value=\"2\""));
        assert!(restored.contains("NEXT-DRAFT"));
        assert_eq!(recorder.freezes.load(Ordering::SeqCst), 3);
        task.abort();
    }

    #[tokio::test]
    async fn editing_an_answer_invalidates_the_old_rating_form() {
        let (base, backend, recorder, task) = serve_test_app(Verdict::Pass).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;
        let edited = client
            .post(format!("{base}/back"))
            .form(&[("session_id", "1"), ("token", "1"), ("instance_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(edited.contains("name=\"token\" value=\"2\""));
        client
            .post(format!("{base}/answer"))
            .form(&[
                ("session_id", "1"),
                ("token", "2"),
                ("instance_id", "1"),
                ("response", "EDITED-RESPONSE"),
            ])
            .send()
            .await
            .unwrap();
        let stale = client
            .post(format!("{base}/rate"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("grade", "good"),
            ])
            .send()
            .await
            .unwrap();
        assert_eq!(stale.status(), StatusCode::CONFLICT);
        assert_eq!(backend.evaluations.load(Ordering::SeqCst), 0);
        assert_eq!(recorder.ungraded_attempts.load(Ordering::SeqCst), 0);
        task.abort();
    }

    #[tokio::test]
    async fn finalization_failure_is_visible_and_retryable() {
        let (base, _backend, recorder, task) = serve_test_app(Verdict::Pass).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;
        rate(&client, &base, "good").await;
        recorder.fail_finishes.store(true, Ordering::SeqCst);

        let failed = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(failed.contains("FINALIZATION ERROR"));
        assert!(failed.contains("FINISH-SENTINEL"));
        assert!(!failed.contains("SESSION COMPLETE"));
        assert!(!failed.contains("Download transcript"));
        assert!(!failed.contains("href=\"session.md\""));
        assert_eq!(recorder.finishes.load(Ordering::SeqCst), 1);
        assert_eq!(recorder.history_queries.load(Ordering::SeqCst), 0);

        recorder.fail_finishes.store(false, Ordering::SeqCst);
        let finished = client
            .post(format!("{base}/retry-finish"))
            .form(&[("session_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("SESSION COMPLETE"));
        assert!(!finished.contains("FINALIZATION ERROR"));
        assert!(finished.contains("Download transcript"));
        assert!(finished.contains("href=\"session.md\" download"));
        assert_eq!(recorder.finishes.load(Ordering::SeqCst), 2);
        assert_eq!(recorder.history_queries.load(Ordering::SeqCst), 1);
        task.abort();
    }

    #[tokio::test]
    async fn shutdown_is_available_only_after_completion_and_stops_the_server() {
        let (base, _backend, recorder, task) = serve_test_app(Verdict::Pass).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();

        let premature = client
            .post(format!("{base}/shutdown"))
            .form(&[("session_id", "1")])
            .send()
            .await
            .unwrap();
        assert_eq!(premature.status(), StatusCode::CONFLICT);

        submit_answer(&client, &base).await;
        rate(&client, &base, "good").await;
        client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap();

        let stopped = client
            .post(format!("{base}/shutdown"))
            .form(&[("session_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(stopped.contains("SHUTTING DOWN"));
        assert!(stopped.contains("SESSION SAVED"));
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("server did not stop")
            .unwrap();
        assert_eq!(recorder.finishes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn archive_uses_authored_directives_not_the_provider_model_name() {
        let output = tempfile::tempdir().unwrap();
        let generated_at = test_timestamp("2026-07-31T12:01:00.000");
        let reviewed_at = test_timestamp("2026-07-31T12:02:00.000");
        let reviews = vec![
            completed_review(
                parsed_spec("What is {{a small multiplication problem}}?"),
                "static",
                1,
                generated_at,
                reviewed_at,
            ),
            completed_review(
                parsed_spec("What is three times four?"),
                "remote-provider",
                2,
                generated_at,
                reviewed_at,
            ),
        ];

        assert!(archive_completed_reviews(output.path(), 1, &reviews).is_none());
        let files = std::fs::read_dir(output.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("md"))
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 1, "the truly static spec must be skipped");
        let archived = std::fs::read_to_string(&files[0]).unwrap();
        assert!(archived.contains("model = \"static\""));
        assert!(archived.contains("Q:\nGenerated question 1"));
        assert!(!archived.contains("Generated question 2"));
    }

    #[tokio::test]
    async fn partial_archive_failure_seals_undo_and_shutdown_retry_is_idempotent() {
        let output = tempfile::tempdir().unwrap();
        let generated_at = test_timestamp("2026-07-31T12:01:00.000");
        let valid_reviewed_at = test_timestamp("2026-07-31T12:02:00.000");
        let invalid_reviewed_at = test_timestamp("2026-07-31T12:00:00.000");
        let reviews = vec![
            completed_review(
                parsed_spec("Calculate {{a two-digit sum}}."),
                "provider",
                1,
                generated_at,
                valid_reviewed_at,
            ),
            completed_review(
                parsed_spec("Translate {{a short phrase}} into French."),
                "provider",
                2,
                generated_at,
                invalid_reviewed_at,
            ),
        ];
        let (state, recorder, shutdown_rx) =
            completed_test_state(reviews, Some(output.path().to_path_buf()));

        let failed = shutdown(
            State(state.clone()),
            Form(SessionForm {
                csrf_token: String::new(),
                session_id: 1,
            }),
        )
        .await;
        assert_eq!(failed.status(), StatusCode::INTERNAL_SERVER_ERROR);
        {
            let mutable = state.mutable.lock().unwrap();
            assert_eq!(mutable.archive_publication, ArchivePublication::Failed);
            assert!(mutable.archive_error.is_some());
            assert!(
                !finished_page(&mutable, "")
                    .into_string()
                    .contains("Undo previous")
            );
        }

        let rejected_undo = undo(
            State(state.clone()),
            Form(UndoForm {
                csrf_token: String::new(),
                session_id: 1,
                review_id: 2,
                current_token: None,
                current_instance_id: None,
                current_draft: None,
            }),
        )
        .await;
        assert_eq!(rejected_undo.status(), StatusCode::CONFLICT);
        assert_eq!(recorder.undos.load(Ordering::SeqCst), 0);

        {
            let mut mutable = state.mutable.lock().unwrap();
            mutable.completed_reviews[1].reviewed_at = valid_reviewed_at;
        }
        let retried = shutdown(
            State(state.clone()),
            Form(SessionForm {
                csrf_token: String::new(),
                session_id: 1,
            }),
        )
        .await;
        assert_eq!(retried.status(), StatusCode::OK);
        tokio::time::timeout(std::time::Duration::from_secs(1), shutdown_rx)
            .await
            .expect("retry did not send shutdown")
            .expect("shutdown sender was dropped");
        assert_eq!(
            state.mutable.lock().unwrap().archive_publication,
            ArchivePublication::Published
        );
        let files = std::fs::read_dir(output.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("md"))
            .count();
        assert_eq!(
            files, 2,
            "retry must not duplicate the first published file"
        );
    }

    #[tokio::test]
    async fn no_archive_destination_keeps_prepublication_undo_available() {
        let review = completed_review(
            parsed_spec("Calculate {{a small product}}."),
            "provider",
            1,
            test_timestamp("2026-07-31T12:01:00.000"),
            test_timestamp("2026-07-31T12:02:00.000"),
        );
        let (state, recorder, _shutdown_rx) = completed_test_state(vec![review], None);

        archive_session(&state).unwrap();
        assert_eq!(
            state.mutable.lock().unwrap().archive_publication,
            ArchivePublication::NotStarted
        );
        let response = undo(
            State(state),
            Form(UndoForm {
                csrf_token: String::new(),
                session_id: 1,
                review_id: 1,
                current_token: None,
                current_instance_id: None,
                current_draft: None,
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert_eq!(recorder.undos.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn completion_stats_distinguish_reviews_from_skips() {
        let mut first = spec();
        first.deck_name = "FIRST-DECK-SENTINEL".into();
        let mut second_spec = parsed_spec("Second authored question");
        second_spec.deck_name = "SECOND-DECK-SENTINEL".into();
        let specs = vec![first, second_spec];
        let (base, _backend, _recorder, task) =
            serve_test_app_with_specs(Verdict::Pass, specs).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;
        rate(&client, &base, "good").await;
        let second = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(second.contains("name=\"token\" value=\"2\""));

        let finished = client
            .post(format!("{base}/skip"))
            .form(&[("session_id", "1"), ("token", "2"), ("instance_id", "2")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("<dt>RESOLVED</dt><dd>2</dd>"));
        assert!(finished.contains("<dt>REVIEWED</dt><dd>1</dd>"));
        assert!(finished.contains("<dt>SKIPPED</dt><dd>1</dd>"));
        assert!(finished.contains("<dt>RETENTION</dt><dd>100%</dd>"));

        let first_transcript = client
            .get(format!("{base}/session.md"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(
            first_transcript.find("FIRST-DECK-SENTINEL").unwrap()
                < first_transcript.find("SECOND-DECK-SENTINEL").unwrap()
        );

        let replacement = client
            .post(format!("{base}/undo"))
            .form(&[("session_id", "1"), ("review_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(replacement.contains("name=\"token\" value=\"3\""));
        let reopened = client
            .get(format!("{base}/session.md"))
            .send()
            .await
            .unwrap();
        assert_eq!(reopened.status(), StatusCode::CONFLICT);

        let completed_again = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "3"),
                ("instance_id", "1"),
                ("action", "regrade"),
                ("grade", "hard"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(completed_again.contains("SESSION COMPLETE"));

        let effective = client
            .get(format!("{base}/session.md"))
            .send()
            .await
            .unwrap();
        assert_eq!(effective.status(), StatusCode::OK);
        let disposition = effective.headers()[CONTENT_DISPOSITION]
            .to_str()
            .unwrap()
            .to_string();
        assert!(disposition.starts_with("attachment; filename=\"hashdrills-session-"));
        let effective = effective.text().await.unwrap();
        assert_eq!(effective.matches("- Status: reviewed").count(), 1);
        assert_eq!(
            effective.matches("- Status: skipped before answer").count(),
            1
        );
        assert!(
            effective.find("SECOND-DECK-SENTINEL").unwrap()
                < effective.find("FIRST-DECK-SENTINEL").unwrap()
        );
        let entries = effective.split("\n## Entry ").skip(1).collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].contains("SECOND-DECK-SENTINEL"));
        assert!(entries[0].contains("- Learner answer: not submitted"));
        assert!(!entries[0].contains("RESPONSE-SENTINEL"));
        assert!(entries[1].contains("FIRST-DECK-SENTINEL"));
        assert!(entries[1].contains("RESPONSE-SENTINEL"));
        assert!(entries[1].contains("- Self-rating: good"));
        assert!(entries[1].contains("- Effective rating: hard"));
        assert!(entries[1].contains("- Resolution: undo regrade"));
        task.abort();
    }

    #[test]
    fn pending_finalization_never_renders_complete() {
        let backend = Arc::new(FakeBackend::new(Verdict::Pass));
        let recorder = Arc::new(FakeRecorder::default());
        let state = AppState::new(
            AccessControl::disabled_for_tests(),
            1,
            Vec::new(),
            Vec::new(),
            std::env::temp_dir(),
            None,
            backend,
            recorder,
            Timestamp::now(),
            None,
        );
        let markup = {
            let mut mutable = state.mutable.lock().unwrap();
            mutable.finish_in_progress = true;
            page(
                &mutable,
                &state.collection_root,
                state.access.csrf_token(),
                state.access.base_path(),
            )
            .into_string()
        };
        assert!(markup.contains("FINALIZING"));
        assert!(!markup.contains("SESSION COMPLETE"));
        assert!(!markup.contains("Download transcript"));
        assert!(!markup.contains("session.md"));
    }

    #[tokio::test]
    async fn skipped_drills_are_resolved_not_reviewed() {
        let (base, _backend, recorder, task) = serve_test_app(Verdict::Pass).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        let finished = client
            .post(format!("{base}/skip"))
            .form(&[("session_id", "1"), ("token", "1"), ("instance_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("SESSION COMPLETE"));
        assert!(finished.contains("<dt>RESOLVED</dt><dd>1</dd>"));
        assert!(finished.contains("<dt>REVIEWED</dt><dd>0</dd>"));
        assert!(finished.contains("<dt>SKIPPED</dt><dd>1</dd>"));
        assert!(finished.contains("<dt>RETENTION</dt><dd>—</dd>"));
        assert_eq!(recorder.graded_attempts.load(Ordering::SeqCst), 0);
        assert!(SCRIPT.contains("undoForm.dataset.submitting"));
        task.abort();
    }

    #[tokio::test]
    async fn unresolved_skip_preserves_answer_and_ai_context_without_a_final_grade() {
        let (base, _backend, recorder, task) = serve_test_app(Verdict::Uncertain).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;
        let unresolved = rate(&client, &base, "easy").await;
        assert!(unresolved.contains("Skip"));

        let finished = client
            .post(format!("{base}/skip"))
            .form(&[("session_id", "1"), ("token", "1"), ("instance_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("SESSION COMPLETE"));
        assert_eq!(recorder.graded_attempts.load(Ordering::SeqCst), 0);

        let transcript = client
            .get(format!("{base}/session.md"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(transcript.contains("- Status: skipped after unresolved check"));
        assert!(transcript.contains("- Self-rating: easy"));
        assert!(transcript.contains("- Effective rating: none"));
        assert!(transcript.contains("- AI verdict: uncertain"));
        assert!(transcript.contains("RESPONSE-SENTINEL"));
        assert!(transcript.contains("FEEDBACK-SENTINEL"));
        task.abort();
    }

    #[tokio::test]
    async fn failed_check_defaults_to_forgot_but_allows_click_only_override() {
        let (base, _backend, recorder, task) = serve_test_app(Verdict::Fail).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;

        let result = rate(&client, &base, "easy").await;
        assert!(result.contains("NAY"));
        assert!(result.contains(">×</span>"));
        assert!(!result.contains(">Fail<"));
        assert!(result.contains("FEEDBACK-SENTINEL"));
        assert!(!result.contains("EVIDENCE-SENTINEL"));
        assert!(result.contains("Keep my rating"));
        assert!(result.contains("Continue as Forgot"));
        assert!(!result.contains("DEFAULT"));
        assert!(!result.contains("<strong>forgot</strong>"));
        assert!(!SCRIPT.contains("Keep my rating"));

        let finished = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "override"),
                ("grade", "easy"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("SESSION COMPLETE"));
        assert!(finished.contains("<dt>RETENTION</dt><dd>100%</dd>"));
        assert_eq!(
            recorder.grades.lock().unwrap()[0],
            (Grade::Easy, ReviewResolution::UserOverride)
        );
        task.abort();
    }

    #[tokio::test]
    async fn uncertain_verdict_never_schedules_and_regenerates_with_new_token() {
        let (base, backend, recorder, task) = serve_test_app(Verdict::Uncertain).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;
        let result = rate(&client, &base, "good").await;
        assert!(result.contains("UNCERTAIN"));
        assert!(result.contains("No schedule update"));
        assert!(result.contains("Regenerate"));
        assert!(!result.contains(">Continue</button>"));
        assert!(!result.contains("Keep my rating"));
        assert_eq!(recorder.graded_attempts.load(Ordering::SeqCst), 0);

        let rejected = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);

        let fresh = client
            .post(format!("{base}/regenerate"))
            .form(&[("session_id", "1"), ("token", "1"), ("instance_id", "1")])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(fresh.contains("QUESTION-SENTINEL"));
        assert!(!fresh.contains("TARGET-SENTINEL"));
        assert!(fresh.contains("value=\"2\""));
        assert_eq!(backend.generations.load(Ordering::SeqCst), 2);

        let stale = client
            .post(format!("{base}/answer"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("response", "stale"),
            ])
            .send()
            .await
            .unwrap();
        assert_eq!(stale.status(), StatusCode::CONFLICT);
        task.abort();
    }

    #[tokio::test]
    async fn failed_check_continue_records_ai_rejected_forgot() {
        let (base, _backend, recorder, task) = serve_test_app(Verdict::Fail).await;
        let client = reqwest::Client::new();
        client.get(format!("{base}/")).send().await.unwrap();
        submit_answer(&client, &base).await;

        let result = rate(&client, &base, "easy").await;
        assert!(result.contains("Continue as Forgot"));
        assert!(!result.contains("DEFAULT"));

        let finished = client
            .post(format!("{base}/accept"))
            .form(&[
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(finished.contains("SESSION COMPLETE"));
        assert!(finished.contains("<dt>RETENTION</dt><dd>0%</dd>"));
        assert_eq!(
            recorder.grades.lock().unwrap()[0],
            (Grade::Forgot, ReviewResolution::AiRejected)
        );
        task.abort();
    }

    #[test]
    fn session_transcript_fences_untrusted_markdown_and_is_not_schedulable() {
        let mut source_spec = spec();
        source_spec.deck_name = "# deck\nQ: injected deck field".into();
        source_spec.goal = Some("---\nA: injected goal\n<img src=x>".into());
        source_spec.source = Some("https://remote.invalid/private.png".into());
        source_spec.path = PathBuf::from("/ABSOLUTE-PATH-MUST-NOT-LEAK/Practice.md");
        let mut review = completed_review(
            source_spec,
            "generation-model",
            1,
            test_timestamp("2026-07-31T12:00:00.000"),
            test_timestamp("2026-07-31T12:00:05.000"),
        );
        review.result.rated.submitted.frozen.instance.question =
            "---\n# injected heading\nQ: fake\nA: fake\n`````\n<img src=x>\n![](https://remote.invalid/x.png)".into();
        review.result.rated.submitted.response =
            "+++\nQ: learner injection\n<script>bad()</script>\n\tTAB\0\u{001B}\u{007F}\u{0085}"
                .into();
        let CheckResult::Evaluation(evaluation) = &mut review.result.check else {
            panic!("fixture has an evaluation");
        };
        evaluation.feedback = "# feedback\nA: injected feedback".into();

        let (state, _recorder, _shutdown) = completed_test_state(vec![review], None);
        let document = {
            let session = state.mutable.lock().unwrap();
            render_session_transcript(&session, session.finished_at.unwrap()).unwrap()
        };
        assert!(document.starts_with(
            "+++\nhashdrills_session_log_kind = \"session_transcript\"\nhashdrills_session_log_format_version = 1\n+++\n"
        ));
        assert!(document.contains("\n``````text\n---\n# injected heading"));
        assert!(document.contains("<script>bad()</script>"));
        assert!(document.contains("\n\tTAB\\u{0000}\\u{001B}\\u{007F}\\u{0085}"));
        assert!(!document.contains('\0'));
        assert!(!document.contains('\u{001B}'));
        assert!(!document.contains('\u{007F}'));
        assert!(!document.contains('\u{0085}'));
        assert!(document.contains("- Generation protocol version: 2"));
        assert!(document.contains("- AI protocol version: 2"));
        assert!(!document.contains("ABSOLUTE-PATH-MUST-NOT-LEAK"));
        assert!(!document.contains("csrf_token"));
        assert!(!document.contains("access_token"));
        assert!(!document.contains("reasoning_effort"));

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("downloaded-session.md");
        write(&path, &document).unwrap();
        assert!(parse_path(&path).unwrap().is_empty());
    }

    #[test]
    fn generation_failure_transcript_never_invents_a_question() {
        let failed_spec = parsed_spec("AUTHORED-TEMPLATE-MUST-NOT-BECOME-A-QUESTION");
        let (state, _recorder, _shutdown) = completed_test_state(Vec::new(), None);
        let document = {
            let mut session = state.mutable.lock().unwrap();
            session.total = 1;
            session.completed = 1;
            let skipped_at = session.finished_at.unwrap();
            session
                .transcript_entries
                .push(SessionTranscriptEntry::Skipped(
                    SkippedDrill::GenerationFailed {
                        spec: failed_spec,
                        skipped_at,
                    },
                ));
            render_session_transcript(&session, session.finished_at.unwrap()).unwrap()
        };
        assert!(document.contains("- Status: skipped after generation failure"));
        assert!(document.contains("- Question: unavailable (generation failed)"));
        assert!(!document.contains("AUTHORED-TEMPLATE-MUST-NOT-BECOME-A-QUESTION"));
    }

    #[test]
    fn transcript_builder_enforces_its_public_size_limit() {
        let mut output = TranscriptWriter::new();
        let maximum = "x".repeat(MAX_SESSION_TRANSCRIPT_BYTES);
        output.push(&maximum).unwrap();
        assert_eq!(output.document.len(), MAX_SESSION_TRANSCRIPT_BYTES);
        assert_eq!(output.push("x"), Err(TranscriptTooLarge));
    }

    #[test]
    fn transcript_readiness_rejects_undo_and_shutdown_races() {
        let review = completed_review(
            spec(),
            "model",
            1,
            test_timestamp("2026-07-31T12:00:00.000"),
            test_timestamp("2026-07-31T12:00:05.000"),
        );
        let (state, _recorder, _shutdown) = completed_test_state(vec![review], None);
        let mut session = state.mutable.lock().unwrap();
        assert!(session_transcript_is_ready(&session));
        session.undo_in_progress = true;
        assert!(!session_transcript_is_ready(&session));
        let markup = finished_page(&session, state.access.csrf_token()).into_string();
        assert!(!markup.contains("Download transcript"));
        assert!(!markup.contains("session.md"));
        session.undo_in_progress = false;
        session.shutdown_requested = true;
        assert!(!session_transcript_is_ready(&session));
    }

    #[tokio::test]
    async fn authenticated_launch_exchanges_the_bearer_and_protects_every_route() {
        let app = serve_security_test_app(true).await;
        let client = redirectless_client();

        let unauthorized = client.get(&app.root_url).send().await.unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(app.backend.generations.load(Ordering::SeqCst), 0);
        assert_eq!(app.recorder.freezes.load(Ordering::SeqCst), 0);

        for path in [
            "/assets/macros.json".to_string(),
            static_assets::KATEX_CSS_URL.to_string(),
            "/file/image.png".to_string(),
            "/session.md".to_string(),
        ] {
            let response = client
                .get(format!("{}{}{}", app.origin, app.base_path, path))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
        }

        let bootstrap = client.get(&app.launch_url).send().await.unwrap();
        assert_eq!(bootstrap.status(), StatusCode::SEE_OTHER);
        assert_eq!(
            bootstrap.headers().get(LOCATION).unwrap().to_str().unwrap(),
            format!("{}/", app.base_path)
        );
        let set_cookie = bootstrap
            .headers()
            .get(SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(set_cookie.contains(&format!("; Path={}/", app.base_path)));
        assert!(set_cookie.contains("; HttpOnly"));
        assert!(set_cookie.contains("; SameSite=Strict"));
        assert!(!set_cookie.contains("; Domain="));
        assert!(!set_cookie.contains("; Secure"));
        assert_eq!(
            bootstrap
                .headers()
                .get(CACHE_CONTROL)
                .unwrap()
                .to_str()
                .unwrap(),
            "no-store"
        );
        assert_eq!(
            bootstrap
                .headers()
                .get(REFERRER_POLICY)
                .unwrap()
                .to_str()
                .unwrap(),
            "strict-origin"
        );
        assert!(bootstrap.headers().contains_key(CONTENT_SECURITY_POLICY));
        let cookie = set_cookie.split(';').next().unwrap().to_string();
        let location = bootstrap
            .headers()
            .get(LOCATION)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let bootstrap_body = bootstrap.text().await.unwrap();
        assert!(!location.contains("access_token"));
        assert!(!bootstrap_body.contains("access_token"));

        let duplicated_cookie = client
            .get(&app.root_url)
            .header(reqwest::header::COOKIE, format!("{cookie}; {cookie}"))
            .send()
            .await
            .unwrap();
        assert_eq!(duplicated_cookie.status(), StatusCode::UNAUTHORIZED);

        let wrong_host = client
            .get(&app.root_url)
            .header(reqwest::header::COOKIE, &cookie)
            .header(reqwest::header::HOST, "attacker.example")
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_host.status(), StatusCode::MISDIRECTED_REQUEST);

        let question = client
            .get(&app.root_url)
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap();
        let question_status = question.status();
        assert_eq!(
            question.headers()[REFERRER_POLICY],
            HeaderValue::from_static("strict-origin")
        );
        let question = question.text().await.unwrap();
        assert_eq!(
            question_status,
            StatusCode::OK,
            "GET {} returned {question}",
            app.root_url
        );
        assert!(question.contains("QUESTION-SENTINEL"));
        assert!(question.contains("action=\"answer\""));
        assert!(question.contains("href=\"assets/"));
        assert!(question.contains(&format!("src=\"{}/file/image.png\"", app.base_path)));
        assert!(question.contains(&format!("name=\"csrf_token\" value=\"{}\"", app.csrf_token)));
        assert_eq!(app.backend.generations.load(Ordering::SeqCst), 1);
        assert_eq!(app.recorder.freezes.load(Ordering::SeqCst), 1);

        let early_transcript = client
            .get(format!("{}session.md", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap();
        assert_eq!(early_transcript.status(), StatusCode::CONFLICT);
        assert_eq!(
            early_transcript.headers()[reqwest::header::CACHE_CONTROL],
            "no-store"
        );

        let macros = client
            .get(format!(
                "{}{}{}",
                app.origin, app.base_path, "/assets/macros.json"
            ))
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap();
        assert_eq!(macros.status(), StatusCode::OK);
        let stylesheet = client
            .get(format!(
                "{}{}{}",
                app.origin,
                app.base_path,
                static_assets::KATEX_CSS_URL
            ))
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap();
        assert_eq!(stylesheet.status(), StatusCode::OK);
        let stylesheet = stylesheet.text().await.unwrap();
        assert!(stylesheet.contains("url(fonts/"));
        assert!(!stylesheet.contains("url(/assets/katex/fonts/"));
        let font = client
            .get(format!(
                "{}{}/assets/katex/fonts/KaTeX_Main-Regular.woff2",
                app.origin, app.base_path
            ))
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap();
        assert_eq!(font.status(), StatusCode::OK);
        let media = client
            .get(format!("{}{}/file/image.png", app.origin, app.base_path))
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap();
        assert_eq!(media.status(), StatusCode::OK);
        assert_eq!(media.headers()[reqwest::header::ACCEPT_RANGES], "bytes");
        assert_eq!(
            media.headers()[reqwest::header::CACHE_CONTROL],
            "private, no-store"
        );
        assert_eq!(media.bytes().await.unwrap().as_ref(), b"PNG-SENTINEL");

        let media_range = client
            .get(format!("{}{}/file/image.png", app.origin, app.base_path))
            .header(reqwest::header::COOKIE, &cookie)
            .header(reqwest::header::RANGE, "bytes=4-")
            .send()
            .await
            .unwrap();
        assert_eq!(media_range.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            media_range.headers()[reqwest::header::CONTENT_RANGE],
            "bytes 4-11/12"
        );
        assert_eq!(media_range.bytes().await.unwrap().as_ref(), b"SENTINEL");

        for path in [
            "/assets/macros.json",
            static_assets::KATEX_CSS_URL,
            "/assets/katex/fonts/KaTeX_Main-Regular.woff2",
            "/file/image.png",
            "/session.md",
        ] {
            let response = client
                .get(format!("{}{}", app.origin, path))
                .header(reqwest::header::COOKIE, &cookie)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }

        let answer_form = [
            ("csrf_token", app.csrf_token.as_str()),
            ("session_id", "1"),
            ("token", "1"),
            ("instance_id", "1"),
            ("response", "RESPONSE-SENTINEL"),
        ];
        let missing_origin = client
            .post(format!("{}answer", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .form(&answer_form)
            .send()
            .await
            .unwrap();
        assert_eq!(missing_origin.status(), StatusCode::FORBIDDEN);

        let null_origin = client
            .post(format!("{}answer", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .header(reqwest::header::ORIGIN, "null")
            .form(&answer_form)
            .send()
            .await
            .unwrap();
        assert_eq!(null_origin.status(), StatusCode::FORBIDDEN);

        let wrong_csrf_form = [
            ("csrf_token", "wrong"),
            ("session_id", "1"),
            ("token", "1"),
            ("instance_id", "1"),
            ("response", "RESPONSE-SENTINEL"),
        ];
        let wrong_csrf = client
            .post(format!("{}answer", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .header(reqwest::header::ORIGIN, &app.origin)
            .form(&wrong_csrf_form)
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_csrf.status(), StatusCode::FORBIDDEN);

        let answered = client
            .post(format!("{}answer", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .header(reqwest::header::ORIGIN, &app.origin)
            .form(&answer_form)
            .send()
            .await
            .unwrap();
        assert_eq!(answered.status(), StatusCode::SEE_OTHER);
        assert_eq!(
            answered.headers().get(LOCATION).unwrap().to_str().unwrap(),
            format!("{}/", app.base_path)
        );
        let rating = client
            .get(&app.root_url)
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(rating.contains("RESPONSE-SENTINEL"));
        assert!(rating.contains("value=\"forgot\""));

        let rated = client
            .post(format!("{}rate", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .header(reqwest::header::ORIGIN, &app.origin)
            .form(&[
                ("csrf_token", app.csrf_token.as_str()),
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("grade", "good"),
            ])
            .send()
            .await
            .unwrap();
        assert_eq!(rated.status(), StatusCode::SEE_OTHER);
        let accepted = client
            .post(format!("{}accept", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .header(reqwest::header::ORIGIN, &app.origin)
            .form(&[
                ("csrf_token", app.csrf_token.as_str()),
                ("session_id", "1"),
                ("token", "1"),
                ("instance_id", "1"),
                ("action", "default"),
            ])
            .send()
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::SEE_OTHER);

        let completion = client
            .get(&app.root_url)
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(completion.contains("href=\"session.md\" download"));
        assert!(!completion.contains("href=\"/session.md\""));

        let transcript = client
            .get(format!("{}session.md", app.root_url))
            .header(reqwest::header::COOKIE, &cookie)
            .send()
            .await
            .unwrap();
        assert_eq!(transcript.status(), StatusCode::OK);
        assert_eq!(
            transcript.headers()[reqwest::header::CONTENT_TYPE],
            "text/markdown; charset=utf-8"
        );
        assert_eq!(
            transcript.headers()[reqwest::header::CACHE_CONTROL],
            "private, no-store"
        );
        assert_eq!(
            transcript.headers()[reqwest::header::X_CONTENT_TYPE_OPTIONS],
            "nosniff"
        );
        let disposition = transcript.headers()[reqwest::header::CONTENT_DISPOSITION]
            .to_str()
            .unwrap()
            .to_string();
        assert!(disposition.starts_with("attachment; filename=\"hashdrills-session-"));
        assert!(disposition.ends_with(".md\""));
        assert!(!disposition.contains("Practice"));
        let transcript = transcript.text().await.unwrap();
        assert!(transcript.contains("RESPONSE-SENTINEL"));
        assert!(transcript.contains("QUESTION-SENTINEL"));
        assert!(transcript.contains("- Resolution: AI confirmed"));
        assert!(!transcript.contains(&app.csrf_token));
        assert!(!transcript.contains("access_token"));
        assert!(!transcript.contains(&app._directory.path().display().to_string()));

        app.task.abort();
    }

    #[tokio::test]
    async fn no_auth_is_loopback_only_but_keeps_host_origin_and_csrf_checks() {
        let app = serve_security_test_app(false).await;
        let client = redirectless_client();
        assert_eq!(app.launch_url, app.root_url);

        let question = client.get(&app.root_url).send().await.unwrap();
        assert_eq!(question.status(), StatusCode::OK);
        assert!(question.headers().get(SET_COOKIE).is_none());
        assert!(question.text().await.unwrap().contains("QUESTION-SENTINEL"));

        let wrong_host = client
            .get(&app.root_url)
            .header(reqwest::header::HOST, "attacker.example")
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_host.status(), StatusCode::MISDIRECTED_REQUEST);

        let answer_form = [
            ("csrf_token", app.csrf_token.as_str()),
            ("session_id", "1"),
            ("token", "1"),
            ("instance_id", "1"),
            ("response", "RESPONSE-SENTINEL"),
        ];
        let missing_origin = client
            .post(format!("{}answer", app.root_url))
            .form(&answer_form)
            .send()
            .await
            .unwrap();
        assert_eq!(missing_origin.status(), StatusCode::FORBIDDEN);

        let wrong_csrf_form = [
            ("csrf_token", "wrong"),
            ("session_id", "1"),
            ("token", "1"),
            ("instance_id", "1"),
            ("response", "RESPONSE-SENTINEL"),
        ];
        let wrong_csrf = client
            .post(format!("{}answer", app.root_url))
            .header(reqwest::header::ORIGIN, &app.origin)
            .form(&wrong_csrf_form)
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_csrf.status(), StatusCode::FORBIDDEN);

        let answered = client
            .post(format!("{}answer", app.root_url))
            .header(reqwest::header::ORIGIN, &app.origin)
            .form(&answer_form)
            .send()
            .await
            .unwrap();
        assert_eq!(answered.status(), StatusCode::SEE_OTHER);
        assert_eq!(
            answered.headers().get(LOCATION).unwrap().to_str().unwrap(),
            format!("{}/", app.base_path)
        );

        app.task.abort();
    }

    #[test]
    fn review_shortcuts_are_global_and_no_timer_remains() {
        assert!(SCRIPT.contains("['1', '2', '3', '4'].includes(key)"));
        assert!(SCRIPT.contains("const grades = ['forgot', 'hard', 'good', 'easy'];"));
        assert!(SCRIPT.contains("gradeForm.querySelector"));
        assert!(!SCRIPT.contains("gradeScope"));
        assert!(SCRIPT.contains("if (key === 'u')"));
        assert!(SCRIPT.contains("defaultForm.requestSubmit()"));

        assert!(!SCRIPT.contains("setInterval"));
        assert!(!SCRIPT.contains("data-auto-advance"));
        assert!(!SCRIPT.contains("data-countdown"));
        assert!(!SCRIPT.contains("data-pause-timer"));
    }

    #[test]
    fn visual_hierarchy_uses_soft_structure_and_verdana() {
        assert!(STYLES.contains("font-family: Verdana, Geneva, sans-serif"));
        assert!(STYLES.contains("--line: #aaa9a1"));
        assert!(STYLES.contains("border: 1px solid var(--line-strong)"));
        assert!(STYLES.contains("background: var(--soft)"));
        assert!(STYLES.contains(".edit-answer { margin-top: 1.75rem; }"));
        assert!(STYLES.contains(".deck { display: inline-block"));
        assert!(STYLES.contains(".goal { margin: 0 0 1.4rem"));
        assert!(STYLES.contains(".question.question-long"));
        assert!(STYLES.contains(".question.question-very-long"));
        assert!(STYLES.contains(
            ".result-question { margin-bottom: .8rem; padding-bottom: 0; border-bottom: 0; }"
        ));
        assert!(STYLES.contains("#9be9a8"));
        assert!(STYLES.contains("#40c463"));
        assert!(STYLES.contains("#30a14e"));
        assert!(STYLES.contains("#216e39"));
        assert!(!STYLES.contains(".card { background: #fff; border: 2px"));
        assert!(!STYLES.contains("outline: 3px solid #111"));
    }

    #[test]
    fn question_typography_scales_with_visible_length() {
        assert_eq!(question_classes(&"x".repeat(240)), "question rich-text");
        assert_eq!(
            question_classes(&"x".repeat(241)),
            "question question-long rich-text"
        );
        assert_eq!(
            question_classes(&format!("{}   \n\t", "x".repeat(500))),
            "question question-long rich-text"
        );
        assert_eq!(
            question_classes(&"x".repeat(501)),
            "question question-very-long rich-text"
        );
    }

    #[test]
    fn deck_header_exposes_a_safe_source_link() {
        let mut drill = spec();
        drill.source = Some("https://example.com/reference".to_string());
        let markup = deck_header(&drill).into_string();
        assert!(markup.contains("class=\"source\""));
        assert!(markup.contains("href=\"https://example.com/reference\""));
        assert!(markup.contains("target=\"_blank\""));
        assert!(markup.contains("rel=\"noopener noreferrer\""));
        assert!(markup.contains("Source ↗"));
    }

    #[test]
    fn duration_and_retention_stats_are_explicit() {
        let start = Timestamp::try_from("2026-07-31T09:00:00.000".to_string()).unwrap();
        let end = Timestamp::try_from("2026-07-31T10:01:01.000".to_string()).unwrap();
        assert_eq!(format_duration(start, end), "1:01:01");
        assert_eq!(format_average_duration(start, end, 2), "30:31");
        assert_eq!(format_average_duration(start, end, 0), "—");
        assert_eq!(format_retention(2, 3), "67%");
        assert_eq!(format_retention(0, 0), "—");
    }

    #[test]
    fn macros_are_loaded_only_from_a_small_allowlisted_file() {
        let directory = tempfile::tempdir().unwrap();
        write(
            directory.path().join("macros.tex"),
            concat!(
                "% comment\n",
                "NOT-A-MACRO SECRET-SENTINEL\n",
                "\\R \\mathbb{R}\n",
                "\\pair #1 + #2\n",
                "\\bad_name should-not-load\n",
                "\\bad1 should-not-load\n",
                "\\control bad\u{7}value\n",
            ),
        )
        .unwrap();

        let macros = load_macros(directory.path());
        assert_eq!(macros.len(), 2);
        assert_eq!(macros.get("\\R").map(String::as_str), Some("\\mathbb{R}"));
        assert_eq!(macros.get("\\pair").map(String::as_str), Some("#1 + #2"));
        assert!(
            !macros
                .values()
                .any(|value| value.contains("SECRET-SENTINEL"))
        );

        write(
            directory.path().join("macros.tex"),
            vec![b'x'; MAX_MACROS_BYTES as usize + 1],
        )
        .unwrap();
        assert!(load_macros(directory.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_macros_file_is_rejected() {
        use std::os::unix::fs::symlink;

        let collection = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        write(&secret, "\\leak SECRET-SENTINEL\n").unwrap();
        symlink(&secret, collection.path().join("macros.tex")).unwrap();

        assert!(load_macros(collection.path()).is_empty());
    }

    #[tokio::test]
    async fn macros_are_delivered_as_non_executable_same_origin_json() {
        let directory = tempfile::tempdir().unwrap();
        write(
            directory.path().join("Practice.md"),
            "Q: Authored question\nA: Authored target\n",
        )
        .unwrap();
        write(
            directory.path().join("macros.tex"),
            "NOT-A-MACRO SECRET-SENTINEL\n\\R \\mathbb{R}\n",
        )
        .unwrap();
        let specs = parse_path(directory.path()).unwrap();
        let backend = Arc::new(FakeBackend::new(Verdict::Pass));
        let recorder = Arc::new(FakeRecorder::default());
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let state = AppState::new(
            AccessControl::disabled_for_tests(),
            1,
            specs.clone(),
            specs,
            directory.path().to_path_buf(),
            None,
            backend,
            recorder,
            Timestamp::now(),
            Some(shutdown_tx),
        );
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router(state))
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        let client = reqwest::Client::new();

        let response = client
            .get(format!("http://{address}/assets/macros.json"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            "application/json; charset=utf-8"
        );
        assert_eq!(
            response.headers().get(X_CONTENT_TYPE_OPTIONS).unwrap(),
            "nosniff"
        );
        assert_eq!(
            response.headers().get(CACHE_CONTROL).unwrap(),
            "private, no-store"
        );
        assert_eq!(
            response
                .headers()
                .get("cross-origin-resource-policy")
                .unwrap(),
            "same-origin"
        );
        assert!(
            response
                .headers()
                .get("access-control-allow-origin")
                .is_none()
        );
        let body = response.text().await.unwrap();
        assert_eq!(
            serde_json::from_str::<HashMap<String, String>>(&body)
                .unwrap()
                .get("\\R")
                .map(String::as_str),
            Some("\\mathbb{R}")
        );
        assert!(!body.contains("SECRET-SENTINEL"));

        let old_script = client
            .get(format!("http://{address}/assets/runtime.js"))
            .send()
            .await
            .unwrap();
        assert_eq!(old_script.status(), StatusCode::NOT_FOUND);
        assert!(SCRIPT.contains("fetch('assets/macros.json'"));
        assert!(!SCRIPT.contains("HASHDRILLS_MACROS"));
        // KaTeX allows \gdef to mutate the provided macro object. Each render
        // must receive a fresh copy so one generated expression cannot affect
        // a later question or criterion.
        assert!(SCRIPT.contains("macros: { ...authoredMacros }"));
        task.abort();
    }
}
