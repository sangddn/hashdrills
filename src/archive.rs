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

//! Optional plaintext archives of reviewed generated drills.
//!
//! Runtime instances remain ephemeral at the filesystem boundary unless a
//! caller explicitly invokes [`save_generated_drill`]. The audit database is
//! authoritative either way. Each accepted review is archived in its own
//! immutable Markdown file so its generation metadata and resolved `G/Q/A`
//! contract cannot become detached from one another.
//!
//! Archive files are evidence, not new schedulable specs. Collection discovery
//! ignores Markdown only when its frontmatter contains the exact
//! [`ARCHIVE_KIND_MARKER_KEY`] and [`ARCHIVE_VERSION_MARKER_KEY`] pair. The CLI
//! also rejects an archive directory nested beneath the collection.
//! Archives preserve generated Markdown, including links, media, and raw HTML.
//! Open them only in a renderer that sanitizes HTML and controls remote loads.
//! They also contain goals, source provenance, activity timestamps, and local
//! trace IDs, so treat the output directory as private unless reviewed before
//! publication. Filenames themselves contain the review date, logical deck
//! name, and local trace IDs. On Unix, Hashdrills creates a newly requested
//! final archive directory with mode `0700`; permissions on an existing
//! directory remain the user's responsibility.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use tempfile::Builder;

use crate::error::ErrorReport;
use crate::model::GeneratedInstance;
use crate::spec::DrillSpec;
use crate::types::timestamp::Timestamp;

/// Version of the plaintext archive format.
pub const ARCHIVE_FORMAT_VERSION: u32 = 1;

/// Archive kind stored beside [`ARCHIVE_FORMAT_VERSION`] in every export.
pub const ARCHIVE_KIND: &str = "generated_drill";

/// Frontmatter key that identifies Hashdrills-generated evidence.
pub const ARCHIVE_KIND_MARKER_KEY: &str = "hashdrills_archive_kind";

/// Frontmatter key that selects the archive representation version.
pub const ARCHIVE_VERSION_MARKER_KEY: &str = "hashdrills_archive_format_version";

const MAX_COLLISION_ATTEMPTS: u32 = 10_000;
const MAX_SLUG_BYTES: usize = 64;

/// Durable identifiers connecting one archive to Hashdrills' local audit DB.
///
/// All four identifiers are required because archives are written only after
/// an accepted review. They are provenance, not scheduler identity; the
/// content-addressed spec hash is stored separately in the file metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveTrace {
    pub session_id: i64,
    pub instance_id: i64,
    pub attempt_id: i64,
    pub review_id: i64,
}

/// Everything needed to archive one fully resolved, accepted drill instance.
///
/// This deliberately accepts neither model prompts nor provider
/// configuration, so those values cannot accidentally enter an archive.
#[derive(Clone, Copy, Debug)]
pub struct ArchivedDrill<'a> {
    pub spec: &'a DrillSpec,
    pub instance: &'a GeneratedInstance,
    /// When the generated instance was frozen, before the learner answered.
    pub generated_at: Timestamp,
    /// When the final scheduler grade was accepted.
    pub reviewed_at: Timestamp,
    pub trace: ArchiveTrace,
}

/// Result of an archive operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SavedDrill {
    pub path: PathBuf,
    /// `false` means an identical archive already existed at this path.
    pub created: bool,
}

/// A validation, serialization, or filesystem failure while archiving.
#[derive(Debug)]
pub enum ArchiveError {
    InvalidRecord(&'static str),
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    UnsafeDestination {
        path: PathBuf,
        reason: &'static str,
    },
    CollisionLimit(PathBuf),
}

impl ArchiveError {
    fn io(operation: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.into(),
            source,
        }
    }
}

impl Display for ArchiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRecord(detail) => write!(formatter, "invalid archive record: {detail}"),
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "could not {operation} {}: {source}",
                path.display()
            ),
            Self::UnsafeDestination { path, reason } => {
                write!(
                    formatter,
                    "unsafe archive directory {}: {reason}",
                    path.display()
                )
            }
            Self::CollisionLimit(path) => write!(
                formatter,
                "could not choose an unused archive filename after {MAX_COLLISION_ATTEMPTS} attempts beneath {}",
                path.display()
            ),
        }
    }
}

impl Error for ArchiveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidRecord(_) | Self::UnsafeDestination { .. } | Self::CollisionLimit(_) => {
                None
            }
        }
    }
}

impl From<ArchiveError> for ErrorReport {
    fn from(error: ArchiveError) -> Self {
        Self::new(format!("archive: {error}"))
    }
}

/// Render one archive file without touching the filesystem.
///
/// The output is UTF-8 Markdown with TOML frontmatter followed by the same
/// optional `G:`, required `Q:`, and required `A:` shape as an authored drill.
/// `resolved_*` metadata values are the canonical, byte-exact strings. The
/// body is a readable projection with each value copied verbatim after its
/// field marker; archive-marked files are excluded from spec discovery.
pub fn render_generated_drill(record: &ArchivedDrill<'_>) -> Result<String, ArchiveError> {
    validate_record(record)?;

    let metadata = render_metadata(record);

    let mut output = String::with_capacity(
        metadata.len()
            + record.instance.question.len()
            + record.instance.target.len()
            + record.instance.rubric.len()
            + record.spec.goal.as_deref().map_or(0, str::len)
            + 64,
    );
    output.push_str("+++\n");
    output.push_str(&metadata);
    if !metadata.ends_with('\n') {
        output.push('\n');
    }
    output.push_str("+++\n\n");

    if let Some(goal) = &record.spec.goal {
        push_exact_field(&mut output, "G", goal);
    }
    push_exact_field(&mut output, "Q", &record.instance.question);
    push_exact_field(&mut output, "A", &record.instance.target);
    Ok(output)
}

fn render_metadata(record: &ArchivedDrill<'_>) -> String {
    // Do not delegate string style to the generic TOML serializer: it may use
    // a multiline string, allowing an adversarial logical name or model ID
    // containing `\n+++\n` to imitate the frontmatter delimiter. Basic strings
    // with explicit escapes keep every metadata value on one physical line.
    let mut metadata = String::new();
    push_toml_string_field(&mut metadata, ARCHIVE_KIND_MARKER_KEY, ARCHIVE_KIND);
    metadata.push_str(&format!(
        "{ARCHIVE_VERSION_MARKER_KEY} = {ARCHIVE_FORMAT_VERSION}\n"
    ));
    push_toml_string_field(&mut metadata, "name", &record.spec.deck_name);
    if let Some(source) = &record.spec.source {
        push_toml_string_field(&mut metadata, "source", source);
    }
    push_toml_string_field(
        &mut metadata,
        "generated_at",
        &record.generated_at.to_string(),
    );
    push_toml_string_field(
        &mut metadata,
        "reviewed_at",
        &record.reviewed_at.to_string(),
    );
    push_toml_string_field(&mut metadata, "spec_hash", &record.spec.hash().to_hex());
    metadata.push_str(&format!("session_id = {}\n", record.trace.session_id));
    metadata.push_str(&format!("instance_id = {}\n", record.trace.instance_id));
    metadata.push_str(&format!("attempt_id = {}\n", record.trace.attempt_id));
    metadata.push_str(&format!("review_id = {}\n", record.trace.review_id));
    push_toml_string_field(&mut metadata, "model", &record.instance.model);
    metadata.push_str(&format!(
        "model_protocol_version = {}\n",
        record.instance.protocol_version
    ));
    if let Some(goal) = &record.spec.goal {
        push_toml_string_field(&mut metadata, "resolved_goal", goal);
    }
    push_toml_string_field(
        &mut metadata,
        "resolved_question",
        &record.instance.question,
    );
    push_toml_string_field(&mut metadata, "resolved_target", &record.instance.target);
    push_toml_string_field(&mut metadata, "resolved_rubric", &record.instance.rubric);
    push_toml_string_field(&mut metadata, "body_projection", "verbatim_gqa_v1");
    push_toml_string_field(
        &mut metadata,
        "content_trust",
        "untrusted_generated_markdown",
    );
    metadata
}

fn push_toml_string_field(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push_str(" = \"");
    for character in value.chars() {
        match character {
            '\u{0008}' => output.push_str("\\b"),
            '\t' => output.push_str("\\t"),
            '\n' => output.push_str("\\n"),
            '\u{000C}' => output.push_str("\\f"),
            '\r' => output.push_str("\\r"),
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04X}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push_str("\"\n");
}

/// Validate and prepare an opt-in plaintext archive directory.
///
/// Call this once during CLI startup, before opening the database or starting
/// the server. It creates the final directory when needed, applies the same
/// privacy and symlink policy as [`save_generated_drill`], durably syncs each
/// newly created directory entry through its pre-existing parent, verifies
/// that a private owned temporary file can be written and synced, removes that
/// probe, syncs the directory entry removal, and returns the canonical
/// destination. Existing directory permissions are never changed.
pub fn prepare_archive_directory(output_dir: impl AsRef<Path>) -> Result<PathBuf, ArchiveError> {
    prepare_archive_directory_with_durability(output_dir.as_ref(), &SystemDurability)
}

fn prepare_archive_directory_with_durability(
    output_dir: &Path,
    durability: &impl Durability,
) -> Result<PathBuf, ArchiveError> {
    let output_dir = prepare_output_directory(output_dir, durability)?;
    let mut probe = Builder::new()
        .prefix(".hashdrills-preflight-")
        .suffix(".tmp")
        .tempfile_in(&output_dir)
        .map_err(|error| ArchiveError::io("create archive preflight file", &output_dir, error))?;
    let probe_path = probe.path().to_path_buf();
    let probe_result = (|| {
        protect_temporary_file(probe.as_file(), &probe_path)?;
        probe
            .write_all(b"hashdrills archive preflight\n")
            .map_err(|error| {
                ArchiveError::io("write archive preflight file", &probe_path, error)
            })?;
        durability
            .sync_temporary(probe.as_file())
            .map_err(|error| ArchiveError::io("sync archive preflight file", &probe_path, error))
    })();

    let cleanup_result = probe
        .close()
        .map_err(|error| ArchiveError::io("remove archive preflight file", &probe_path, error));
    let directory_sync_result = durability.sync_directory(&output_dir).map_err(|error| {
        ArchiveError::io("sync archive directory after preflight", &output_dir, error)
    });

    // Always attempt cleanup and its durability barrier, even when the probe
    // write or file sync failed. Cleanup failures take precedence because they
    // may leave an artifact that the preflight promises not to retain.
    cleanup_result?;
    directory_sync_result?;
    probe_result?;
    Ok(output_dir)
}

/// Atomically create a plaintext archive beneath `output_dir`.
///
/// The destination is never overwritten. A byte-identical pre-existing file
/// makes the call idempotent; a different file with the same natural name gets
/// a numeric suffix. Temporary files are created in the destination directory
/// and atomically persisted with no-clobber semantics, so concurrent writers
/// cannot expose a partial file or overwrite one another.
pub fn save_generated_drill(
    output_dir: impl AsRef<Path>,
    record: &ArchivedDrill<'_>,
) -> Result<SavedDrill, ArchiveError> {
    save_generated_drill_with_durability(output_dir.as_ref(), record, &SystemDurability)
}

trait Durability {
    fn sync_temporary(&self, file: &File) -> io::Result<()>;
    fn sync_published(&self, file: &File) -> io::Result<()>;
    fn sync_existing(&self, file: &File) -> io::Result<()>;
    fn sync_directory(&self, path: &Path) -> io::Result<()>;
}

struct SystemDurability;

impl Durability for SystemDurability {
    fn sync_temporary(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }

    fn sync_published(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }

    fn sync_existing(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }

    #[cfg(unix)]
    fn sync_directory(&self, path: &Path) -> io::Result<()> {
        File::open(path)?.sync_all()
    }

    #[cfg(not(unix))]
    fn sync_directory(&self, _path: &Path) -> io::Result<()> {
        Ok(())
    }
}

fn save_generated_drill_with_durability(
    output_dir: &Path,
    record: &ArchivedDrill<'_>,
    durability: &impl Durability,
) -> Result<SavedDrill, ArchiveError> {
    let rendered = render_generated_drill(record)?;
    let output_dir = prepare_output_directory(output_dir, durability)?;

    let mut temporary = Builder::new()
        .prefix(".hashdrills-")
        .suffix(".tmp")
        .tempfile_in(&output_dir)
        .map_err(|error| ArchiveError::io("create archive temporary file", &output_dir, error))?;
    protect_temporary_file(temporary.as_file(), temporary.path())?;
    temporary.write_all(rendered.as_bytes()).map_err(|error| {
        ArchiveError::io("write archive temporary file", temporary.path(), error)
    })?;
    durability
        .sync_temporary(temporary.as_file())
        .map_err(|error| {
            ArchiveError::io("sync archive temporary file", temporary.path(), error)
        })?;

    let base = archive_filename_base(record);
    for collision in 0..MAX_COLLISION_ATTEMPTS {
        let filename = collision_filename(&base, collision);
        let destination = output_dir.join(filename);
        match temporary.persist_noclobber(&destination) {
            Ok(file) => {
                durability
                    .sync_published(&file)
                    .map_err(|error| ArchiveError::io("sync archive file", &destination, error))?;
                durability.sync_directory(&output_dir).map_err(|error| {
                    ArchiveError::io("sync archive directory", &output_dir, error)
                })?;
                return Ok(SavedDrill {
                    path: destination,
                    created: true,
                });
            }
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
                temporary = error.file;
                if sync_if_file_equals(&destination, rendered.as_bytes(), &output_dir, durability)?
                {
                    return Ok(SavedDrill {
                        path: destination,
                        created: false,
                    });
                }
            }
            Err(error) => {
                let source = error.error;
                drop(error.file);
                return Err(ArchiveError::io(
                    "persist archive file",
                    destination,
                    source,
                ));
            }
        }
    }

    Err(ArchiveError::CollisionLimit(output_dir))
}

fn prepare_output_directory(
    path: &Path,
    durability: &impl Durability,
) -> Result<PathBuf, ArchiveError> {
    // An explicitly symlinked final directory is surprising for an operation
    // that promises archives "beneath" the requested folder, so reject it.
    // Symlinked ancestors (including macOS' conventional `/tmp`) are resolved
    // by canonicalization and all subsequent operations use that resolved path.
    let mut existing_ancestor = None;
    let created = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(ArchiveError::UnsafeDestination {
                path: path.to_path_buf(),
                reason: "the final path is a symbolic link",
            });
        }
        Ok(metadata) if !metadata.is_dir() => {
            return Err(ArchiveError::UnsafeDestination {
                path: path.to_path_buf(),
                reason: "the path exists but is not a directory",
            });
        }
        Ok(_) => false,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            existing_ancestor = Some(nearest_existing_ancestor(path)?);
            fs::create_dir_all(path)
                .map_err(|error| ArchiveError::io("create archive directory", path, error))?;
            true
        }
        Err(error) => {
            return Err(ArchiveError::io("inspect archive directory", path, error));
        }
    };

    let metadata = fs::symlink_metadata(path)
        .map_err(|error| ArchiveError::io("inspect archive directory", path, error))?;
    if metadata.file_type().is_symlink() {
        return Err(ArchiveError::UnsafeDestination {
            path: path.to_path_buf(),
            reason: "the final path became a symbolic link",
        });
    }
    if !metadata.is_dir() {
        return Err(ArchiveError::UnsafeDestination {
            path: path.to_path_buf(),
            reason: "the path is not a directory",
        });
    }
    let resolved = path
        .canonicalize()
        .map_err(|error| ArchiveError::io("resolve archive directory", path, error))?;
    if created {
        protect_new_directory(&resolved)?;
        sync_created_directory_chain(
            &resolved,
            existing_ancestor
                .as_deref()
                .ok_or_else(|| ArchiveError::UnsafeDestination {
                    path: path.to_path_buf(),
                    reason: "could not identify the existing parent directory",
                })?,
            durability,
        )?;
    }
    Ok(resolved)
}

fn nearest_existing_ancestor(path: &Path) -> Result<PathBuf, ArchiveError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| ArchiveError::io("resolve current directory", path, error))?
            .join(path)
    };
    let mut candidate = absolute.parent();
    while let Some(ancestor) = candidate {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => {
                return ancestor.canonicalize().map_err(|error| {
                    ArchiveError::io("resolve archive parent directory", ancestor, error)
                });
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                candidate = ancestor.parent();
            }
            Err(error) => {
                return Err(ArchiveError::io(
                    "inspect archive parent directory",
                    ancestor,
                    error,
                ));
            }
        }
    }
    Err(ArchiveError::UnsafeDestination {
        path: path.to_path_buf(),
        reason: "the path has no existing parent directory",
    })
}

fn sync_created_directory_chain(
    resolved: &Path,
    existing_ancestor: &Path,
    durability: &impl Durability,
) -> Result<(), ArchiveError> {
    if !resolved.starts_with(existing_ancestor) {
        return Err(ArchiveError::UnsafeDestination {
            path: resolved.to_path_buf(),
            reason: "the created directory resolved outside its inspected parent",
        });
    }

    for directory in resolved.ancestors() {
        durability.sync_directory(directory).map_err(|error| {
            ArchiveError::io("sync new archive directory hierarchy", directory, error)
        })?;
        if directory == existing_ancestor {
            return Ok(());
        }
    }

    Err(ArchiveError::UnsafeDestination {
        path: resolved.to_path_buf(),
        reason: "the created directory is detached from its inspected parent",
    })
}

#[cfg(unix)]
fn protect_new_directory(path: &Path) -> Result<(), ArchiveError> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)
        .map_err(|error| ArchiveError::io("inspect new archive directory", path, error))?
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions)
        .map_err(|error| ArchiveError::io("protect new archive directory", path, error))
}

#[cfg(unix)]
fn protect_temporary_file(file: &File, path: &Path) -> Result<(), ArchiveError> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = file
        .metadata()
        .map_err(|error| ArchiveError::io("inspect archive temporary file", path, error))?
        .permissions();
    permissions.set_mode(0o600);
    file.set_permissions(permissions)
        .map_err(|error| ArchiveError::io("protect archive temporary file", path, error))
}

#[cfg(not(unix))]
fn protect_new_directory(_path: &Path) -> Result<(), ArchiveError> {
    Ok(())
}

#[cfg(not(unix))]
fn protect_temporary_file(_file: &File, _path: &Path) -> Result<(), ArchiveError> {
    Ok(())
}

fn validate_record(record: &ArchivedDrill<'_>) -> Result<(), ArchiveError> {
    if record.spec.deck_name.trim().is_empty() {
        return Err(ArchiveError::InvalidRecord(
            "logical name must not be blank",
        ));
    }
    if record.instance.question.trim().is_empty() {
        return Err(ArchiveError::InvalidRecord("question must not be blank"));
    }
    if record.instance.target.trim().is_empty() {
        return Err(ArchiveError::InvalidRecord("target must not be blank"));
    }
    if record.instance.rubric.trim().is_empty() {
        return Err(ArchiveError::InvalidRecord("rubric must not be blank"));
    }
    if record.instance.model.trim().is_empty() {
        return Err(ArchiveError::InvalidRecord("model must not be blank"));
    }
    if record.instance.protocol_version == 0 {
        return Err(ArchiveError::InvalidRecord(
            "model protocol version must be positive",
        ));
    }
    if [
        record.trace.session_id,
        record.trace.instance_id,
        record.trace.attempt_id,
        record.trace.review_id,
    ]
    .iter()
    .any(|identifier| *identifier <= 0)
    {
        return Err(ArchiveError::InvalidRecord(
            "trace identifiers must be positive",
        ));
    }
    if record.reviewed_at.into_inner() < record.generated_at.into_inner() {
        return Err(ArchiveError::InvalidRecord(
            "review date precedes generation date",
        ));
    }
    Ok(())
}

fn push_exact_field(output: &mut String, tag: &str, content: &str) {
    output.push_str(tag);
    output.push_str(":\n");
    output.push_str(content);
    if !content.ends_with('\n') {
        output.push('\n');
    }
    output.push('\n');
}

fn archive_filename_base(record: &ArchivedDrill<'_>) -> String {
    format!(
        "{}-{}-s{}-i{}-r{}",
        record.reviewed_at.date(),
        safe_slug(&record.spec.deck_name),
        record.trace.session_id,
        record.trace.instance_id,
        record.trace.review_id
    )
}

fn collision_filename(base: &str, collision: u32) -> String {
    if collision == 0 {
        format!("{base}.md")
    } else {
        format!("{base}-{}.md", collision + 1)
    }
}

fn safe_slug(name: &str) -> String {
    let mut slug = String::with_capacity(name.len().min(MAX_SLUG_BYTES));
    let mut needs_separator = false;
    for character in name.chars() {
        let normalized = character.to_ascii_lowercase();
        if normalized.is_ascii_alphanumeric() {
            if needs_separator && !slug.is_empty() && slug.len() < MAX_SLUG_BYTES {
                slug.push('-');
            }
            needs_separator = false;
            if slug.len() < MAX_SLUG_BYTES {
                slug.push(normalized);
            }
        } else {
            needs_separator = !slug.is_empty();
        }
        if slug.len() >= MAX_SLUG_BYTES {
            break;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug.push_str("drill");
    }
    if is_windows_reserved_name(&slug) {
        slug.insert_str(0, "drill-");
    }
    slug
}

fn is_windows_reserved_name(slug: &str) -> bool {
    matches!(slug, "con" | "prn" | "aux" | "nul")
        || slug
            .strip_prefix("com")
            .or_else(|| slug.strip_prefix("lpt"))
            .is_some_and(|suffix| suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9'))
}

fn sync_if_file_equals(
    path: &Path,
    expected: &[u8],
    directory: &Path,
    durability: &impl Durability,
) -> Result<bool, ArchiveError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(ArchiveError::io(
                "inspect existing archive file",
                path,
                error,
            ));
        }
    };
    if !metadata.file_type().is_file()
        || metadata.len() != u64::try_from(expected.len()).unwrap_or(u64::MAX)
    {
        return Ok(false);
    }
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| ArchiveError::io("open existing archive file", path, error))?;
    let mut buffer = [0_u8; 8 * 1024];
    let mut offset = 0;
    while offset < expected.len() {
        let length = buffer.len().min(expected.len() - offset);
        file.read_exact(&mut buffer[..length])
            .map_err(|error| ArchiveError::io("read existing archive file", path, error))?;
        if buffer[..length] != expected[offset..offset + length] {
            return Ok(false);
        }
        offset += length;
    }
    let mut extra = [0_u8; 1];
    let read = file
        .read(&mut extra)
        .map_err(|error| ArchiveError::io("read existing archive file", path, error))?;
    if read != 0 {
        return Ok(false);
    }

    // An identical destination may be residue from an earlier call that
    // published successfully but failed during file or directory fsync. Never
    // report idempotent success until both durability barriers pass now.
    durability
        .sync_existing(&file)
        .map_err(|error| ArchiveError::io("sync existing archive file", path, error))?;
    durability
        .sync_directory(directory)
        .map_err(|error| ArchiveError::io("sync existing archive directory", directory, error))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use chrono::NaiveDateTime;
    use tempfile::tempdir;

    use super::*;
    use crate::model::PROTOCOL_VERSION;
    use crate::spec::{Template, parse_path};

    type TestResult = Result<(), Box<dyn Error>>;

    fn timestamp(value: &str) -> Timestamp {
        Timestamp::new(NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.3f").unwrap())
    }

    fn fixture_spec(name: &str, source: Option<&str>) -> DrillSpec {
        DrillSpec::new(
            name,
            source.map(str::to_string),
            PathBuf::from("Decks/Source.md"),
            (4, 6),
            Some("Practice the underlying operation.".into()),
            Template::parse("What is {{two times six}}?").unwrap(),
            Template::parse("{{the exact result}}").unwrap(),
        )
    }

    fn fixture_instance(question: &str) -> GeneratedInstance {
        GeneratedInstance {
            question: question.into(),
            target: "12".into(),
            rubric: "12".into(),
            model: "openai/gpt-test".into(),
            protocol_version: PROTOCOL_VERSION,
        }
    }

    fn fixture_record<'a>(
        spec: &'a DrillSpec,
        instance: &'a GeneratedInstance,
    ) -> ArchivedDrill<'a> {
        ArchivedDrill {
            spec,
            instance,
            generated_at: timestamp("2026-07-31T12:00:00.123"),
            reviewed_at: timestamp("2026-07-31T12:00:08.456"),
            trace: ArchiveTrace {
                session_id: 7,
                instance_id: 11,
                attempt_id: 13,
                review_id: 17,
            },
        }
    }

    fn frontmatter(document: &str) -> &str {
        let body = document.strip_prefix("+++\n").unwrap();
        body.split_once("\n+++\n").unwrap().0
    }

    fn archive_body(document: &str) -> &str {
        let document = document.strip_prefix("+++\n").unwrap();
        document.split_once("\n+++\n\n").unwrap().1
    }

    #[test]
    fn renders_whitelisted_metadata_and_resolved_gqa() -> TestResult {
        let spec = fixture_spec(
            "Bounded multiplication",
            Some("https://example.test/multiplication"),
        );
        let instance = fixture_instance("What is 2 × 6?");
        let rendered = render_generated_drill(&fixture_record(&spec, &instance))?;
        let metadata: toml::Value = toml::from_str(frontmatter(&rendered))?;

        assert_eq!(
            metadata["hashdrills_archive_kind"].as_str(),
            Some("generated_drill")
        );
        assert_eq!(
            metadata["hashdrills_archive_format_version"].as_integer(),
            Some(1)
        );
        assert_eq!(metadata["name"].as_str(), Some("Bounded multiplication"));
        assert_eq!(
            metadata["source"].as_str(),
            Some("https://example.test/multiplication")
        );
        assert_eq!(
            metadata["generated_at"].as_str(),
            Some("2026-07-31T12:00:00.123")
        );
        assert_eq!(
            metadata["reviewed_at"].as_str(),
            Some("2026-07-31T12:00:08.456")
        );
        assert_eq!(metadata["session_id"].as_integer(), Some(7));
        assert_eq!(metadata["instance_id"].as_integer(), Some(11));
        assert_eq!(metadata["attempt_id"].as_integer(), Some(13));
        assert_eq!(metadata["review_id"].as_integer(), Some(17));
        assert_eq!(metadata["model"].as_str(), Some("openai/gpt-test"));
        assert_eq!(
            metadata["resolved_goal"].as_str(),
            Some("Practice the underlying operation.")
        );
        assert_eq!(
            metadata["resolved_question"].as_str(),
            Some("What is 2 × 6?")
        );
        assert_eq!(metadata["resolved_target"].as_str(), Some("12"));
        assert_eq!(metadata["resolved_rubric"].as_str(), Some("12"));
        assert_eq!(
            metadata["body_projection"].as_str(),
            Some("verbatim_gqa_v1")
        );
        assert!(metadata.get("prompt").is_none());
        assert!(metadata.get("api_key").is_none());
        assert_eq!(
            archive_body(&rendered),
            "G:\nPractice the underlying operation.\n\nQ:\nWhat is 2 × 6?\n\nA:\n12\n\n"
        );
        assert!(!rendered.contains("{{two times six}}"));
        assert!(!rendered.contains("{{the exact result}}"));
        Ok(())
    }

    #[test]
    fn toml_and_markdown_delimiters_cannot_break_the_archive_shape() -> TestResult {
        let spec = fixture_spec("Deck \"name\"\n+++\nsource = \"forged\"", None);
        let question = "Inspect `{{ literal }}`.\r\nQ: quoted line\r\n---\r\n```toml\r\n+++\r\n```";
        let target = "Return `}}` literally.\nA: not a field";
        let rubric = "Accept `{{` and `}}` byte-for-byte.\r\n---";
        let mut instance = fixture_instance(question);
        instance.target = target.into();
        instance.rubric = rubric.into();
        instance.model = "provider/\"model\"\n+++".into();

        let rendered = render_generated_drill(&fixture_record(&spec, &instance))?;
        let metadata: toml::Value = toml::from_str(frontmatter(&rendered))?;
        assert_eq!(
            metadata["name"].as_str(),
            Some("Deck \"name\"\n+++\nsource = \"forged\"")
        );
        assert_eq!(metadata["model"].as_str(), Some("provider/\"model\"\n+++"));
        assert_eq!(metadata["resolved_question"].as_str(), Some(question));
        assert_eq!(metadata["resolved_target"].as_str(), Some(target));
        assert_eq!(metadata["resolved_rubric"].as_str(), Some(rubric));
        let body = archive_body(&rendered);
        assert!(body.contains(&format!("Q:\n{question}")));
        assert!(body.contains(&format!("A:\n{target}")));
        assert!(body.contains("\r\n---\r\n"));
        assert!(body.contains("`{{ literal }}`"));
        assert!(body.contains("\r\n+++\r\n"));
        assert!(!body.contains("\\{{"));
        assert!(!body.contains("\n Q: quoted line"));
        assert!(!body.contains("\n***\n"));

        // The marker makes the complete archive non-schedulable even though
        // its verbatim body deliberately contains parser control syntax.
        let directory = tempdir()?;
        let archive = directory.path().join("Archived.md");
        fs::write(&archive, &rendered)?;
        assert!(parse_path(&archive)?.is_empty());
        Ok(())
    }

    #[test]
    fn preserves_markdown_and_whitespace_verbatim_with_canonical_metadata() -> TestResult {
        let spec = fixture_spec("Markdown fidelity", None);
        let question = "  Leading spaces survive.\n\n```text\nQ: literal field\n---\n{{literal braces}}\n```\nUse `x = {{y}}` exactly.\n---\n    indented {{block}}\n";
        let target = "  A precise target.\n\n```md\nA: literal answer\n```\n";
        let rubric = "Allow inline `{{braces}}`; reject normalization.\n---\n";
        let mut instance = fixture_instance(question);
        instance.target = target.into();
        instance.rubric = rubric.into();

        let rendered = render_generated_drill(&fixture_record(&spec, &instance))?;
        let metadata: toml::Value = toml::from_str(frontmatter(&rendered))?;
        assert_eq!(metadata["resolved_question"].as_str(), Some(question));
        assert_eq!(metadata["resolved_target"].as_str(), Some(target));
        assert_eq!(metadata["resolved_rubric"].as_str(), Some(rubric));

        let body = archive_body(&rendered);
        assert!(body.contains(&format!("Q:\n{question}")));
        assert!(body.contains(&format!("A:\n{target}")));
        assert!(body.contains("```text\nQ: literal field\n---\n{{literal braces}}\n```"));
        assert!(body.contains("Use `x = {{y}}` exactly."));
        assert!(body.contains("\n    indented {{block}}\n"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_an_explicitly_symlinked_output_directory() -> TestResult {
        use std::os::unix::fs::symlink;

        let parent = tempdir()?;
        let target = parent.path().join("target");
        fs::create_dir(&target)?;
        let link = parent.path().join("archive-link");
        symlink(&target, &link)?;
        let spec = fixture_spec("Symlink", None);
        let instance = fixture_instance("Question");

        assert!(matches!(
            save_generated_drill(&link, &fixture_record(&spec, &instance)),
            Err(ArchiveError::UnsafeDestination {
                reason: "the final path is a symbolic link",
                ..
            })
        ));
        assert_eq!(fs::read_dir(&target)?.count(), 0);
        Ok(())
    }

    #[test]
    fn uses_safe_filenames_and_never_escapes_the_output_directory() -> TestResult {
        let directory = tempdir()?;
        for name in ["../../AUX", "💥 / ..", "COM1"] {
            let spec = fixture_spec(name, None);
            let instance = fixture_instance("A distinct question");
            let saved = save_generated_drill(directory.path(), &fixture_record(&spec, &instance))?;
            assert_eq!(
                saved.path.parent(),
                Some(directory.path().canonicalize()?.as_path())
            );
            let filename = saved.path.file_name().unwrap().to_str().unwrap();
            assert!(filename.ends_with(".md"));
            assert!(filename.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'.')
            }));
            assert!(!filename.contains(".."));
        }
        Ok(())
    }

    #[test]
    fn identical_save_is_idempotent_and_real_collision_gets_suffix() -> TestResult {
        let directory = tempdir()?;
        let spec = fixture_spec("Collision deck", None);
        let first_instance = fixture_instance("First question");
        let first_record = fixture_record(&spec, &first_instance);

        let first = save_generated_drill(directory.path(), &first_record)?;
        let repeated = save_generated_drill(directory.path(), &first_record)?;
        assert!(first.created);
        assert!(!repeated.created);
        assert_eq!(first.path, repeated.path);

        let second_instance = fixture_instance("Different database, same trace IDs");
        let second =
            save_generated_drill(directory.path(), &fixture_record(&spec, &second_instance))?;
        assert!(second.created);
        assert_ne!(first.path, second.path);
        assert!(
            second
                .path
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with("-2")
        );
        assert_eq!(
            fs::read_to_string(&first.path)?,
            render_generated_drill(&first_record)?
        );
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum InjectedFailure {
        PublishedFile,
        Directory,
    }

    struct FailOnceDurability {
        failure: InjectedFailure,
        failed: AtomicBool,
        published: AtomicBool,
        existing_syncs: AtomicUsize,
        directory_syncs: AtomicUsize,
    }

    impl FailOnceDurability {
        fn new(failure: InjectedFailure) -> Self {
            Self {
                failure,
                failed: AtomicBool::new(false),
                published: AtomicBool::new(false),
                existing_syncs: AtomicUsize::new(0),
                directory_syncs: AtomicUsize::new(0),
            }
        }
    }

    impl Durability for FailOnceDurability {
        fn sync_temporary(&self, file: &File) -> io::Result<()> {
            file.sync_all()
        }

        fn sync_published(&self, file: &File) -> io::Result<()> {
            self.published.store(true, Ordering::SeqCst);
            if matches!(self.failure, InjectedFailure::PublishedFile)
                && !self.failed.swap(true, Ordering::SeqCst)
            {
                Err(io::Error::other("injected published-file sync failure"))
            } else {
                file.sync_all()
            }
        }

        fn sync_existing(&self, file: &File) -> io::Result<()> {
            self.existing_syncs.fetch_add(1, Ordering::SeqCst);
            file.sync_all()
        }

        fn sync_directory(&self, path: &Path) -> io::Result<()> {
            if self.published.load(Ordering::SeqCst) {
                self.directory_syncs.fetch_add(1, Ordering::SeqCst);
                if matches!(self.failure, InjectedFailure::Directory)
                    && !self.failed.swap(true, Ordering::SeqCst)
                {
                    return Err(io::Error::other("injected directory sync failure"));
                }
            }
            #[cfg(unix)]
            {
                File::open(path)?.sync_all()
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                Ok(())
            }
        }
    }

    #[test]
    fn retry_syncs_identical_file_left_by_a_failed_durability_barrier() -> TestResult {
        let parent = tempdir()?;
        let spec = fixture_spec("Durable retry", None);
        let instance = fixture_instance("Will the retry fsync this file?");
        let record = fixture_record(&spec, &instance);
        for (index, failure, expected_operation, expected_directory_syncs) in [
            (0, InjectedFailure::PublishedFile, "sync archive file", 1),
            (1, InjectedFailure::Directory, "sync archive directory", 2),
        ] {
            let directory = parent.path().join(index.to_string());
            let durability = FailOnceDurability::new(failure);

            let first = save_generated_drill_with_durability(&directory, &record, &durability);
            match first {
                Err(ArchiveError::Io { operation, .. }) => {
                    assert_eq!(operation, expected_operation);
                }
                result => panic!("expected injected sync failure, received {result:?}"),
            }
            assert_eq!(
                fs::read_dir(&directory)?
                    .filter_map(Result::ok)
                    .filter(|entry| entry.path().extension().is_some_and(|value| value == "md"))
                    .count(),
                1
            );

            let retried = save_generated_drill_with_durability(&directory, &record, &durability)?;
            assert!(!retried.created);
            assert_eq!(durability.existing_syncs.load(Ordering::SeqCst), 1);
            assert_eq!(
                durability.directory_syncs.load(Ordering::SeqCst),
                expected_directory_syncs
            );
        }
        Ok(())
    }

    struct RecordingDirectoryDurability {
        synced: Mutex<Vec<PathBuf>>,
        fail_on_call: Option<usize>,
        calls: AtomicUsize,
    }

    impl RecordingDirectoryDurability {
        fn new(fail_on_call: Option<usize>) -> Self {
            Self {
                synced: Mutex::new(Vec::new()),
                fail_on_call,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Durability for RecordingDirectoryDurability {
        fn sync_temporary(&self, file: &File) -> io::Result<()> {
            file.sync_all()
        }

        fn sync_published(&self, file: &File) -> io::Result<()> {
            file.sync_all()
        }

        fn sync_existing(&self, file: &File) -> io::Result<()> {
            file.sync_all()
        }

        fn sync_directory(&self, path: &Path) -> io::Result<()> {
            self.synced.lock().unwrap().push(path.to_path_buf());
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_call == Some(call) {
                return Err(io::Error::other(
                    "injected directory hierarchy sync failure",
                ));
            }
            #[cfg(unix)]
            {
                File::open(path)?.sync_all()
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                Ok(())
            }
        }
    }

    #[test]
    fn creating_nested_archive_directory_syncs_each_new_entry_and_its_parent() -> TestResult {
        let parent = tempdir()?;
        let requested = parent.path().join("first").join("second");
        let durability = RecordingDirectoryDurability::new(None);

        let prepared = prepare_output_directory(&requested, &durability)?;

        let synced = durability.synced.lock().unwrap();
        assert_eq!(
            synced.as_slice(),
            [
                prepared,
                requested.parent().unwrap().canonicalize()?,
                parent.path().canonicalize()?,
            ]
        );
        Ok(())
    }

    #[test]
    fn directory_hierarchy_sync_failures_abort_preflight_before_probe_creation() -> TestResult {
        let parent = tempdir()?;
        let requested = parent.path().join("first").join("second");
        let durability = RecordingDirectoryDurability::new(Some(2));

        let error = prepare_archive_directory_with_durability(&requested, &durability).unwrap_err();

        assert!(matches!(
            error,
            ArchiveError::Io {
                operation: "sync new archive directory hierarchy",
                ..
            }
        ));
        assert_eq!(durability.calls.load(Ordering::SeqCst), 2);
        assert!(fs::read_dir(&requested)?.all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".hashdrills-preflight-")
        }));
        Ok(())
    }

    struct FailPreflightFileSync;

    impl Durability for FailPreflightFileSync {
        fn sync_temporary(&self, _file: &File) -> io::Result<()> {
            Err(io::Error::other("injected preflight-file sync failure"))
        }

        fn sync_published(&self, _file: &File) -> io::Result<()> {
            unreachable!("preflight does not publish an archive")
        }

        fn sync_existing(&self, _file: &File) -> io::Result<()> {
            unreachable!("preflight does not inspect an existing archive")
        }

        fn sync_directory(&self, path: &Path) -> io::Result<()> {
            #[cfg(unix)]
            {
                File::open(path)?.sync_all()
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                Ok(())
            }
        }
    }

    #[test]
    fn preflight_creates_a_canonical_empty_directory_and_removes_its_probe() -> TestResult {
        let parent = tempdir()?;
        let requested = parent.path().join("generated-drills");

        let prepared = prepare_archive_directory(&requested)?;

        assert_eq!(prepared, requested.canonicalize()?);
        assert!(prepared.is_dir());
        assert_eq!(fs::read_dir(&prepared)?.count(), 0);
        Ok(())
    }

    #[test]
    fn preflight_removes_its_probe_when_the_writeability_check_fails() -> TestResult {
        let parent = tempdir()?;
        let requested = parent.path().join("generated-drills");

        let error = prepare_archive_directory_with_durability(&requested, &FailPreflightFileSync)
            .unwrap_err();

        assert!(matches!(
            error,
            ArchiveError::Io {
                operation: "sync archive preflight file",
                ..
            }
        ));
        assert_eq!(fs::read_dir(&requested)?.count(), 0);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn newly_created_archive_directory_is_private_but_existing_mode_is_preserved() -> TestResult {
        use std::os::unix::fs::PermissionsExt;

        let parent = tempdir()?;
        let spec = fixture_spec("Private archive", None);
        let instance = fixture_instance("Question");
        let record = fixture_record(&spec, &instance);

        let created = parent.path().join("created");
        let saved = save_generated_drill(&created, &record)?;
        assert_eq!(fs::metadata(&created)?.permissions().mode() & 0o777, 0o700);
        assert_eq!(
            fs::metadata(saved.path)?.permissions().mode() & 0o777,
            0o600
        );

        let existing = parent.path().join("existing");
        fs::create_dir(&existing)?;
        fs::set_permissions(&existing, fs::Permissions::from_mode(0o755))?;
        save_generated_drill(&existing, &record)?;
        assert_eq!(fs::metadata(&existing)?.permissions().mode() & 0o777, 0o755);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn preflight_keeps_new_directories_private_and_existing_modes_unchanged() -> TestResult {
        use std::os::unix::fs::PermissionsExt;

        let parent = tempdir()?;
        let created = parent.path().join("created");
        prepare_archive_directory(&created)?;
        assert_eq!(fs::metadata(&created)?.permissions().mode() & 0o777, 0o700);

        let existing = parent.path().join("existing");
        fs::create_dir(&existing)?;
        fs::set_permissions(&existing, fs::Permissions::from_mode(0o755))?;
        prepare_archive_directory(&existing)?;
        assert_eq!(fs::metadata(&existing)?.permissions().mode() & 0o777, 0o755);
        assert_eq!(fs::read_dir(&existing)?.count(), 0);
        Ok(())
    }

    #[test]
    fn concurrent_identical_writers_create_exactly_one_file() -> TestResult {
        let directory = tempdir()?;
        let spec = Arc::new(fixture_spec("Concurrent deck", None));
        let instance = Arc::new(fixture_instance("What is synchronized?"));

        let results = thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..16 {
                let spec = Arc::clone(&spec);
                let instance = Arc::clone(&instance);
                let output = directory.path().to_path_buf();
                handles.push(scope.spawn(move || {
                    save_generated_drill(output, &fixture_record(&spec, &instance))
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Result<Vec<_>, _>>()
        })?;

        assert_eq!(results.iter().filter(|result| result.created).count(), 1);
        assert_eq!(
            results
                .iter()
                .map(|result| result.path.clone())
                .collect::<HashSet<_>>()
                .len(),
            1
        );
        let entries = fs::read_dir(directory.path())?.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0]
                .path()
                .extension()
                .and_then(|value| value.to_str()),
            Some("md")
        );
        Ok(())
    }

    #[test]
    fn keeps_distinct_target_and_rubric_in_the_answer_contract() -> TestResult {
        let spec = fixture_spec("Rubric", None);
        let mut instance = fixture_instance("Explain the result.");
        instance.target = "The result is twelve.".into();
        instance.rubric = "Must mention multiplication.".into();

        let rendered = render_generated_drill(&fixture_record(&spec, &instance))?;
        let metadata: toml::Value = toml::from_str(frontmatter(&rendered))?;
        assert_eq!(
            metadata["resolved_target"].as_str(),
            Some("The result is twelve.")
        );
        assert_eq!(
            metadata["resolved_rubric"].as_str(),
            Some("Must mention multiplication.")
        );
        assert!(archive_body(&rendered).ends_with("A:\nThe result is twelve.\n\n"));
        assert!(!archive_body(&rendered).contains("Additional rubric"));
        Ok(())
    }

    #[test]
    fn rejects_non_review_records_before_creating_a_directory() {
        let parent = tempdir().unwrap();
        let output = parent.path().join("not-created");
        let spec = fixture_spec("Invalid trace", None);
        let instance = fixture_instance("Question");
        let mut record = fixture_record(&spec, &instance);
        record.trace.review_id = 0;

        assert!(matches!(
            save_generated_drill(&output, &record),
            Err(ArchiveError::InvalidRecord(
                "trace identifiers must be positive"
            ))
        ));
        assert!(!output.exists());
    }

    #[test]
    fn rejects_a_review_date_before_generation() {
        let spec = fixture_spec("Bad chronology", None);
        let instance = fixture_instance("Question");
        let mut record = fixture_record(&spec, &instance);
        record.reviewed_at = timestamp("2026-07-31T11:59:59.999");

        assert!(matches!(
            render_generated_drill(&record),
            Err(ArchiveError::InvalidRecord(
                "review date precedes generation date"
            ))
        ));
    }
}
