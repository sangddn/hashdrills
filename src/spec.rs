// Copyright 2025–2026 Fernando Borretti
//
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

//! Authored drill specifications and their Markdown parser.
//!
//! A [`DrillSpec`] is the stable, scheduled object. Generated questions are
//! instances of it and deliberately do not participate in its identity.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs::read_to_string;
use std::path::Path;
use std::path::PathBuf;

use rusqlite::ToSql;
use rusqlite::types::FromSql;
use rusqlite::types::FromSqlError;
use rusqlite::types::FromSqlResult;
use rusqlite::types::ToSqlOutput;
use rusqlite::types::ValueRef;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use walkdir::WalkDir;

use crate::archive::{ARCHIVE_FORMAT_VERSION, ARCHIVE_KIND};
use crate::error::ErrorReport;
use crate::transcript::{SESSION_LOG_FORMAT_VERSION, SESSION_LOG_KIND};

/// Version of the canonical drill-spec hash encoding.
pub const SPEC_HASH_VERSION: u8 = 1;

const SPEC_HASH_DOMAIN: &[u8] = b"hashdrills:drill-spec";

/// One semantic part of a question or answer template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplatePart {
    /// Text copied to the generated result verbatim.
    Literal(String),
    /// Instructions whose result is supplied by the model backend.
    Directive(String),
}

/// A parsed `{{directive}}` template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Template {
    parts: Vec<TemplatePart>,
    source: String,
}

impl Template {
    /// Parse a template and return a line/column diagnostic on failure.
    pub fn parse(input: &str) -> Result<Self, TemplateParseError> {
        let normalized = normalize_newlines(input);
        let input = normalized.as_ref();
        let mut parts = Vec::new();
        let mut literal = String::new();
        let mut index = 0;
        let mut line = 1;
        let mut column = 1;

        while index < input.len() {
            if input[index..].starts_with("\\{{") {
                literal.push_str("{{");
                advance_ascii(&mut index, &mut column, 3);
            } else if input[index..].starts_with("\\}}") {
                literal.push_str("}}");
                advance_ascii(&mut index, &mut column, 3);
            } else if input[index..].starts_with("{{") {
                if !literal.is_empty() {
                    parts.push(TemplatePart::Literal(std::mem::take(&mut literal)));
                }

                let opening_line = line;
                let opening_column = column;
                advance_ascii(&mut index, &mut column, 2);
                let mut directive = String::new();
                let mut closed = false;

                while index < input.len() {
                    if input[index..].starts_with("\\{{") {
                        directive.push_str("{{");
                        advance_ascii(&mut index, &mut column, 3);
                    } else if input[index..].starts_with("\\}}") {
                        directive.push_str("}}");
                        advance_ascii(&mut index, &mut column, 3);
                    } else if input[index..].starts_with("{{") {
                        return Err(TemplateParseError::new(
                            "directives cannot be nested",
                            line,
                            column,
                        ));
                    } else if input[index..].starts_with("}}") {
                        advance_ascii(&mut index, &mut column, 2);
                        if directive.trim().is_empty() {
                            return Err(TemplateParseError::new(
                                "directive cannot be empty",
                                opening_line,
                                opening_column,
                            ));
                        }
                        parts.push(TemplatePart::Directive(directive.trim().to_string()));
                        closed = true;
                        break;
                    } else {
                        push_next_char(input, &mut index, &mut line, &mut column, &mut directive);
                    }
                }

                if !closed {
                    return Err(TemplateParseError::new(
                        "unclosed directive",
                        opening_line,
                        opening_column,
                    ));
                }
            } else if input[index..].starts_with("}}") {
                return Err(TemplateParseError::new(
                    "closing directive delimiter has no matching opening delimiter",
                    line,
                    column,
                ));
            } else {
                push_next_char(input, &mut index, &mut line, &mut column, &mut literal);
            }
        }

        if !literal.is_empty() {
            parts.push(TemplatePart::Literal(literal));
        }
        trim_outer_literal_whitespace(&mut parts);
        let source = canonical_template_source(&parts);
        Ok(Self { parts, source })
    }

    /// The canonical authored representation, including directive delimiters.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Parsed literal and directive parts, in authored order.
    #[allow(dead_code)]
    pub fn parts(&self) -> &[TemplatePart] {
        &self.parts
    }

    /// The number of replacements required by [`Template::render`].
    pub fn directive_count(&self) -> usize {
        self.parts
            .iter()
            .filter(|part| matches!(part, TemplatePart::Directive(_)))
            .count()
    }

    /// Directive instructions, in the same order expected by `render`.
    pub fn directives(&self) -> impl Iterator<Item = &str> {
        self.parts.iter().filter_map(|part| match part {
            TemplatePart::Literal(_) => None,
            TemplatePart::Directive(instructions) => Some(instructions.as_str()),
        })
    }

    /// Replace every directive with the corresponding generated string.
    pub fn render(&self, replacements: &[String]) -> Result<String, TemplateRenderError> {
        let expected = self.directive_count();
        if replacements.len() != expected {
            return Err(TemplateRenderError {
                expected,
                actual: replacements.len(),
            });
        }

        let mut result = String::new();
        let mut replacement_index = 0;
        for part in &self.parts {
            match part {
                TemplatePart::Literal(text) => result.push_str(text),
                TemplatePart::Directive(_) => {
                    result.push_str(&replacements[replacement_index]);
                    replacement_index += 1;
                }
            }
        }
        Ok(result)
    }
}

impl Display for Template {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source)
    }
}

/// A syntax error in a `{{directive}}` template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl TemplateParseError {
    fn new(message: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            message: message.into(),
            line,
            column,
        }
    }
}

impl Display for TemplateParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} at line {}, column {}",
            self.message, self.line, self.column
        )
    }
}

impl Error for TemplateParseError {}

/// A replacement-count mismatch while rendering a template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemplateRenderError {
    pub expected: usize,
    pub actual: usize,
}

impl Display for TemplateRenderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "template needs {} replacement(s), but {} were provided",
            self.expected, self.actual
        )
    }
}

impl Error for TemplateRenderError {}

/// Content address of a parsed drill specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpecHash {
    inner: blake3::Hash,
}

impl SpecHash {
    /// Lowercase hexadecimal representation used in SQLite and JSON.
    pub fn to_hex(self) -> String {
        self.inner.to_hex().to_string()
    }

    /// Decode a 32-byte BLAKE3 hash from hexadecimal.
    pub fn from_hex(value: &str) -> Result<Self, SpecHashParseError> {
        let inner =
            blake3::Hash::from_hex(value).map_err(|_| SpecHashParseError(value.to_string()))?;
        Ok(Self { inner })
    }

    fn for_content(goal: Option<&str>, question: &Template, answer: &Template) -> Self {
        let mut hasher = blake3::Hasher::new();
        update_hash_bytes(&mut hasher, SPEC_HASH_DOMAIN);
        hasher.update(&[SPEC_HASH_VERSION]);

        match goal {
            Some(goal) => {
                hasher.update(&[1]);
                update_hash_bytes(&mut hasher, goal.as_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        update_hash_template(&mut hasher, question);
        update_hash_template(&mut hasher, answer);
        Self {
            inner: hasher.finalize(),
        }
    }
}

impl PartialOrd for SpecHash {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpecHash {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.as_bytes().cmp(other.inner.as_bytes())
    }
}

impl Display for SpecHash {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Serialize for SpecHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for SpecHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_hex(&value).map_err(serde::de::Error::custom)
    }
}

impl ToSql for SpecHash {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_hex()))
    }
}

impl FromSql for SpecHash {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let value: String = FromSql::column_result(value)?;
        Self::from_hex(&value).map_err(|error| FromSqlError::Other(Box::new(error)))
    }
}

/// Invalid hexadecimal representation of a [`SpecHash`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecHashParseError(String);

impl Display for SpecHashParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid drill spec hash: {}", self.0)
    }
}

impl Error for SpecHashParseError {}

/// One stable, schedulable drill definition parsed from Markdown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrillSpec {
    pub deck_name: String,
    pub source: Option<String>,
    pub path: PathBuf,
    /// One-based, inclusive source line range.
    pub range: (usize, usize),
    pub goal: Option<String>,
    pub question: Template,
    pub answer: Template,
    hash: SpecHash,
}

impl DrillSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        deck_name: impl Into<String>,
        source: Option<String>,
        path: PathBuf,
        range: (usize, usize),
        goal: Option<String>,
        question: Template,
        answer: Template,
    ) -> Self {
        let goal = goal
            .map(|goal| goal.trim().to_string())
            .filter(|goal| !goal.is_empty());
        let hash = SpecHash::for_content(goal.as_deref(), &question, &answer);
        Self {
            deck_name: deck_name.into(),
            source,
            path,
            range,
            goal,
            question,
            answer,
            hash,
        }
    }

    /// Stable identity derived only from the canonical `G/Q/A` content.
    pub fn hash(&self) -> SpecHash {
        self.hash
    }
}

/// A file-aware drill specification diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecError {
    pub message: String,
    pub path: PathBuf,
    pub line: usize,
    pub column: usize,
}

impl SpecError {
    fn new(
        message: impl Into<String>,
        path: impl Into<PathBuf>,
        line: usize,
        column: usize,
    ) -> Self {
        Self {
            message: message.into(),
            path: path.into(),
            line,
            column,
        }
    }
}

impl Display for SpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} Location: {}:{}:{}",
            self.message,
            self.path.display(),
            self.line,
            self.column
        )
    }
}

impl Error for SpecError {}

impl From<SpecError> for ErrorReport {
    fn from(value: SpecError) -> Self {
        ErrorReport::new(value.to_string())
    }
}

/// Parse either one Markdown file or every Markdown file below a directory.
///
/// Files named `._*.md` are ignored. Results are sorted by content hash, and a
/// duplicate hash is an error rather than an implicit de-duplication.
pub fn parse_path(path: impl AsRef<Path>) -> Result<Vec<DrillSpec>, SpecError> {
    let requested = path.as_ref();
    let root = requested.canonicalize().map_err(|error| {
        SpecError::new(
            format!("could not open drill path: {error}"),
            requested,
            1,
            1,
        )
    })?;

    let mut files = Vec::new();
    if root.is_file() {
        if is_appledouble_markdown(&root) {
            return Ok(Vec::new());
        }
        if !is_markdown(&root) {
            return Err(SpecError::new(
                "expected a Markdown (.md) file",
                &root,
                1,
                1,
            ));
        }
        files.push(root.clone());
    } else if root.is_dir() {
        for entry in WalkDir::new(&root) {
            let entry = entry.map_err(|error| {
                let error_path = error.path().unwrap_or(&root);
                SpecError::new(
                    format!("could not traverse drill directory: {error}"),
                    error_path,
                    1,
                    1,
                )
            })?;
            let entry_path = entry.path();
            if entry.file_type().is_file()
                && is_markdown(entry_path)
                && !is_appledouble_markdown(entry_path)
            {
                files.push(entry_path.to_path_buf());
            }
        }
        files.sort();
    } else {
        return Err(SpecError::new(
            "drill path is neither a file nor a directory",
            &root,
            1,
            1,
        ));
    }

    let mut result = Vec::new();
    let mut seen: HashMap<SpecHash, (PathBuf, (usize, usize))> = HashMap::new();
    for file in files {
        for spec in parse_file(&file)? {
            if let Some((first_path, first_range)) = seen.get(&spec.hash()) {
                return Err(SpecError::new(
                    format!(
                        "duplicate drill specification; first defined at {}:{}-{}",
                        first_path.display(),
                        first_range.0,
                        first_range.1
                    ),
                    &spec.path,
                    spec.range.0,
                    1,
                ));
            }
            seen.insert(spec.hash(), (spec.path.clone(), spec.range));
            result.push(spec);
        }
    }
    result.sort_by_key(DrillSpec::hash);
    Ok(result)
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileMetadata {
    name: Option<String>,
    source: Option<String>,
    #[serde(skip)]
    archived: bool,
}

#[derive(Debug, Default, Deserialize)]
struct GeneratedOutputMarker {
    hashdrills_archive_kind: Option<String>,
    hashdrills_archive_format_version: Option<u32>,
    hashdrills_session_log_kind: Option<String>,
    hashdrills_session_log_format_version: Option<u32>,
}

#[derive(Debug)]
struct Field {
    tag_line: usize,
    raw: String,
}

impl Field {
    fn new(tag_line: usize, first_line: &str) -> Self {
        Self {
            tag_line,
            raw: first_line.to_string(),
        }
    }

    fn push_line(&mut self, line: &str) {
        self.raw.push('\n');
        self.raw.push_str(line);
    }
}

enum ParserState {
    Start,
    Goal {
        start_line: usize,
        goal: Field,
    },
    Question {
        start_line: usize,
        goal: Option<String>,
        question: Field,
    },
    Answer {
        start_line: usize,
        goal: Option<String>,
        question: Field,
        answer: Field,
    },
}

#[derive(Clone, Copy)]
enum Tag<'a> {
    Goal(&'a str),
    Question(&'a str),
    Answer(&'a str),
    Separator,
    Text(&'a str),
}

fn parse_file(path: &Path) -> Result<Vec<DrillSpec>, SpecError> {
    let text = read_to_string(path).map_err(|error| {
        SpecError::new(format!("could not read drill file: {error}"), path, 1, 1)
    })?;
    let lines: Vec<&str> = text.lines().collect();
    let (metadata, content_start) = extract_frontmatter(path, &lines)?;
    if metadata.archived {
        return Ok(Vec::new());
    }
    let deck_name = metadata.name.unwrap_or_else(|| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("None")
            .to_string()
    });
    let mut state = ParserState::Start;
    let mut specs = Vec::new();

    for (index, line) in lines.iter().enumerate().skip(content_start) {
        let line_number = index + 1;
        let tag = classify_line(line);
        state = match state {
            ParserState::Start => match tag {
                Tag::Goal(content) => ParserState::Goal {
                    start_line: line_number,
                    goal: Field::new(line_number, content),
                },
                Tag::Question(content) => ParserState::Question {
                    start_line: line_number,
                    goal: None,
                    question: Field::new(line_number, content),
                },
                Tag::Answer(_) => {
                    return Err(SpecError::new(
                        "found A: without an associated Q:",
                        path,
                        line_number,
                        1,
                    ));
                }
                Tag::Separator => ParserState::Start,
                Tag::Text(content) if content.trim().is_empty() => ParserState::Start,
                Tag::Text(_) => {
                    return Err(SpecError::new(
                        "found text outside a drill; expected G: or Q: at column 0",
                        path,
                        line_number,
                        1,
                    ));
                }
            },
            ParserState::Goal {
                start_line,
                mut goal,
            } => match tag {
                Tag::Question(content) => {
                    let goal = parse_goal(path, goal)?;
                    ParserState::Question {
                        start_line,
                        goal: Some(goal),
                        question: Field::new(line_number, content),
                    }
                }
                Tag::Goal(_) => {
                    return Err(SpecError::new(
                        "found a new G: while reading a goal",
                        path,
                        line_number,
                        1,
                    ));
                }
                Tag::Answer(_) => {
                    return Err(SpecError::new("found A: before Q:", path, line_number, 1));
                }
                Tag::Separator => {
                    return Err(SpecError::new(
                        "separator found before goal had a Q: and A:",
                        path,
                        line_number,
                        1,
                    ));
                }
                Tag::Text(content) => {
                    goal.push_line(content);
                    ParserState::Goal { start_line, goal }
                }
            },
            ParserState::Question {
                start_line,
                goal,
                mut question,
            } => match tag {
                Tag::Answer(content) => ParserState::Answer {
                    start_line,
                    goal,
                    question,
                    answer: Field::new(line_number, content),
                },
                Tag::Goal(_) | Tag::Question(_) => {
                    return Err(SpecError::new(
                        "found a new drill before the current Q: had an A:",
                        path,
                        line_number,
                        1,
                    ));
                }
                Tag::Separator => {
                    return Err(SpecError::new(
                        "separator found before Q: had an A:",
                        path,
                        line_number,
                        1,
                    ));
                }
                Tag::Text(content) => {
                    question.push_line(content);
                    ParserState::Question {
                        start_line,
                        goal,
                        question,
                    }
                }
            },
            ParserState::Answer {
                start_line,
                goal,
                question,
                mut answer,
            } => match tag {
                Tag::Goal(content) => {
                    specs.push(finalize_spec(
                        path,
                        &deck_name,
                        metadata.source.as_deref(),
                        start_line,
                        line_number.saturating_sub(1),
                        goal,
                        question,
                        answer,
                    )?);
                    ParserState::Goal {
                        start_line: line_number,
                        goal: Field::new(line_number, content),
                    }
                }
                Tag::Question(content) => {
                    specs.push(finalize_spec(
                        path,
                        &deck_name,
                        metadata.source.as_deref(),
                        start_line,
                        line_number.saturating_sub(1),
                        goal,
                        question,
                        answer,
                    )?);
                    ParserState::Question {
                        start_line: line_number,
                        goal: None,
                        question: Field::new(line_number, content),
                    }
                }
                Tag::Answer(_) => {
                    return Err(SpecError::new(
                        "found a new A: while reading an answer",
                        path,
                        line_number,
                        1,
                    ));
                }
                Tag::Separator => {
                    specs.push(finalize_spec(
                        path,
                        &deck_name,
                        metadata.source.as_deref(),
                        start_line,
                        line_number.saturating_sub(1),
                        goal,
                        question,
                        answer,
                    )?);
                    ParserState::Start
                }
                Tag::Text(content) => {
                    answer.push_line(content);
                    ParserState::Answer {
                        start_line,
                        goal,
                        question,
                        answer,
                    }
                }
            },
        };
    }

    match state {
        ParserState::Start => {}
        ParserState::Goal { goal, .. } => {
            return Err(SpecError::new(
                "file ended before G: had a Q: and A:",
                path,
                goal.tag_line,
                1,
            ));
        }
        ParserState::Question { question, .. } => {
            return Err(SpecError::new(
                "file ended before Q: had an A:",
                path,
                question.tag_line,
                1,
            ));
        }
        ParserState::Answer {
            start_line,
            goal,
            question,
            answer,
        } => specs.push(finalize_spec(
            path,
            &deck_name,
            metadata.source.as_deref(),
            start_line,
            lines.len().max(start_line),
            goal,
            question,
            answer,
        )?),
    }

    if specs.is_empty() {
        return Err(SpecError::new(
            "no drill specifications found in Markdown file",
            path,
            content_start.saturating_add(1),
            1,
        ));
    }
    Ok(specs)
}

#[allow(clippy::too_many_arguments)]
fn finalize_spec(
    path: &Path,
    deck_name: &str,
    source: Option<&str>,
    start_line: usize,
    end_line: usize,
    goal: Option<String>,
    question: Field,
    answer: Field,
) -> Result<DrillSpec, SpecError> {
    let question = parse_template_field(path, "question", question)?;
    let answer = parse_template_field(path, "answer", answer)?;
    Ok(DrillSpec::new(
        deck_name,
        source.map(str::to_string),
        path.to_path_buf(),
        (start_line, end_line.max(start_line)),
        goal,
        question,
        answer,
    ))
}

fn parse_goal(path: &Path, field: Field) -> Result<String, SpecError> {
    let goal = field.raw.trim().to_string();
    if goal.is_empty() {
        return Err(SpecError::new(
            "goal cannot be empty",
            path,
            field.tag_line,
            3,
        ));
    }
    Ok(goal)
}

fn parse_template_field(
    path: &Path,
    field_name: &str,
    field: Field,
) -> Result<Template, SpecError> {
    let template = Template::parse(&field.raw).map_err(|error| {
        let line = field.tag_line + error.line - 1;
        let column = if error.line == 1 {
            error.column + 2
        } else {
            error.column
        };
        SpecError::new(
            format!("invalid {field_name} template: {}", error.message),
            path,
            line,
            column,
        )
    })?;
    if template.source().is_empty() {
        return Err(SpecError::new(
            format!("{field_name} cannot be empty"),
            path,
            field.tag_line,
            3,
        ));
    }
    Ok(template)
}

fn extract_frontmatter(path: &Path, lines: &[&str]) -> Result<(FileMetadata, usize), SpecError> {
    if lines.first().is_none_or(|line| line.trim() != "+++") {
        return Ok((FileMetadata::default(), 0));
    }

    let closing = lines
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(index, line)| (line.trim() == "+++").then_some(index))
        .ok_or_else(|| {
            SpecError::new("frontmatter opening '+++' has no closing '+++'", path, 1, 1)
        })?;
    let source = lines[1..closing].join("\n");
    let marker: GeneratedOutputMarker = toml::from_str(&source).map_err(|error| {
        SpecError::new(
            format!("failed to parse TOML frontmatter: {error}"),
            path,
            2,
            1,
        )
    })?;
    let has_archive_marker = marker.hashdrills_archive_kind.is_some()
        || marker.hashdrills_archive_format_version.is_some();
    let has_session_log_marker = marker.hashdrills_session_log_kind.is_some()
        || marker.hashdrills_session_log_format_version.is_some();
    if has_archive_marker && has_session_log_marker {
        let (line, column) =
            frontmatter_key_location(lines, closing, "hashdrills_session_log_kind");
        return Err(SpecError::new(
            "a file may contain only one Hashdrills generated-output marker family",
            path,
            line,
            column,
        ));
    }
    if has_archive_marker {
        match (
            marker.hashdrills_archive_kind.as_deref(),
            marker.hashdrills_archive_format_version,
        ) {
            (Some(ARCHIVE_KIND), Some(ARCHIVE_FORMAT_VERSION)) => {
                return ignored_generated_output(closing);
            }
            (Some(ARCHIVE_KIND), Some(version)) => {
                let (line, column) =
                    frontmatter_key_location(lines, closing, "hashdrills_archive_format_version");
                return Err(SpecError::new(
                    format!(
                        "unsupported archive format version {version}; supported version is {ARCHIVE_FORMAT_VERSION}"
                    ),
                    path,
                    line,
                    column,
                ));
            }
            (Some(kind), Some(_)) => {
                let (line, column) =
                    frontmatter_key_location(lines, closing, "hashdrills_archive_kind");
                return Err(SpecError::new(
                    format!(
                        "unsupported Hashdrills archive kind {kind:?}; supported kind is {ARCHIVE_KIND:?}"
                    ),
                    path,
                    line,
                    column,
                ));
            }
            (Some(_), None) => {
                let (line, column) =
                    frontmatter_key_location(lines, closing, "hashdrills_archive_kind");
                return Err(SpecError::new(
                    "incomplete Hashdrills archive marker: hashdrills_archive_kind requires hashdrills_archive_format_version",
                    path,
                    line,
                    column,
                ));
            }
            (None, Some(_)) => {
                let (line, column) =
                    frontmatter_key_location(lines, closing, "hashdrills_archive_format_version");
                return Err(SpecError::new(
                    "incomplete Hashdrills archive marker: hashdrills_archive_format_version requires hashdrills_archive_kind",
                    path,
                    line,
                    column,
                ));
            }
            (None, None) => unreachable!("archive marker presence was checked"),
        }
    }
    if has_session_log_marker {
        match (
            marker.hashdrills_session_log_kind.as_deref(),
            marker.hashdrills_session_log_format_version,
        ) {
            (Some(SESSION_LOG_KIND), Some(SESSION_LOG_FORMAT_VERSION)) => {
                return ignored_generated_output(closing);
            }
            (Some(SESSION_LOG_KIND), Some(version)) => {
                let (line, column) = frontmatter_key_location(
                    lines,
                    closing,
                    "hashdrills_session_log_format_version",
                );
                return Err(SpecError::new(
                    format!(
                        "unsupported session transcript format version {version}; supported version is {SESSION_LOG_FORMAT_VERSION}"
                    ),
                    path,
                    line,
                    column,
                ));
            }
            (Some(kind), Some(_)) => {
                let (line, column) =
                    frontmatter_key_location(lines, closing, "hashdrills_session_log_kind");
                return Err(SpecError::new(
                    format!(
                        "unsupported Hashdrills session-log kind {kind:?}; supported kind is {SESSION_LOG_KIND:?}"
                    ),
                    path,
                    line,
                    column,
                ));
            }
            (Some(_), None) => {
                let (line, column) =
                    frontmatter_key_location(lines, closing, "hashdrills_session_log_kind");
                return Err(SpecError::new(
                    "incomplete Hashdrills session-log marker: hashdrills_session_log_kind requires hashdrills_session_log_format_version",
                    path,
                    line,
                    column,
                ));
            }
            (None, Some(_)) => {
                let (line, column) = frontmatter_key_location(
                    lines,
                    closing,
                    "hashdrills_session_log_format_version",
                );
                return Err(SpecError::new(
                    "incomplete Hashdrills session-log marker: hashdrills_session_log_format_version requires hashdrills_session_log_kind",
                    path,
                    line,
                    column,
                ));
            }
            (None, None) => unreachable!("session-log marker presence was checked"),
        }
    }
    let mut metadata: FileMetadata = toml::from_str(&source).map_err(|error| {
        SpecError::new(
            format!("failed to parse TOML frontmatter: {error}"),
            path,
            2,
            1,
        )
    })?;
    let (source_line, source_column) = frontmatter_source_location(lines, closing);
    metadata.source = normalize_source(metadata.source, path, source_line, source_column)?;
    Ok((metadata, closing + 1))
}

fn ignored_generated_output(closing: usize) -> Result<(FileMetadata, usize), SpecError> {
    Ok((
        FileMetadata {
            archived: true,
            ..FileMetadata::default()
        },
        closing + 1,
    ))
}

/// Normalize and validate the optional source link from drill frontmatter.
///
/// This intentionally matches Hashcards: sources may link to the web or a
/// local Obsidian note, while unsafe and ambiguous URI schemes are rejected.
fn normalize_source(
    source: Option<String>,
    path: &Path,
    line: usize,
    column: usize,
) -> Result<Option<String>, SpecError> {
    let Some(source) = source else {
        return Ok(None);
    };
    let source = source.trim();
    if source.is_empty() {
        return Err(SpecError::new(
            "frontmatter source must not be empty",
            path,
            line,
            column,
        ));
    }
    if !source.starts_with("https://")
        && !source.starts_with("http://")
        && !source.starts_with("obsidian://")
    {
        return Err(SpecError::new(
            "frontmatter source must use http://, https://, or obsidian://",
            path,
            line,
            column,
        ));
    }
    Ok(Some(source.to_string()))
}

fn frontmatter_source_location(lines: &[&str], closing: usize) -> (usize, usize) {
    for (index, line) in lines.iter().enumerate().take(closing).skip(1) {
        let trimmed = line.trim_start();
        let is_source_key = ["source", "\"source\"", "'source'"].iter().any(|key| {
            trimmed
                .strip_prefix(key)
                .is_some_and(|rest| rest.trim_start().starts_with('='))
        });
        if is_source_key {
            let indentation = line.len() - trimmed.len();
            return (index + 1, line[..indentation].chars().count() + 1);
        }
    }
    (2, 1)
}

fn frontmatter_key_location(lines: &[&str], closing: usize, key: &str) -> (usize, usize) {
    let double_quoted = format!("\"{key}\"");
    let single_quoted = format!("'{key}'");
    for (index, line) in lines.iter().enumerate().take(closing).skip(1) {
        let trimmed = line.trim_start();
        let is_key = [key, double_quoted.as_str(), single_quoted.as_str()]
            .iter()
            .any(|candidate| {
                trimmed
                    .strip_prefix(candidate)
                    .is_some_and(|rest| rest.trim_start().starts_with('='))
            });
        if is_key {
            let indentation = line.len() - trimmed.len();
            return (index + 1, line[..indentation].chars().count() + 1);
        }
    }
    (2, 1)
}

fn classify_line(line: &str) -> Tag<'_> {
    if let Some(content) = line.strip_prefix("G:") {
        Tag::Goal(content)
    } else if let Some(content) = line.strip_prefix("Q:") {
        Tag::Question(content)
    } else if let Some(content) = line.strip_prefix("A:") {
        Tag::Answer(content)
    } else if line.trim() == "---" {
        Tag::Separator
    } else {
        Tag::Text(line)
    }
}

fn is_markdown(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
}

fn is_appledouble_markdown(path: &Path) -> bool {
    is_markdown(path)
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("._"))
}

fn normalize_newlines(input: &str) -> Cow<'_, str> {
    if input.contains('\r') {
        Cow::Owned(input.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(input)
    }
}

fn advance_ascii(index: &mut usize, column: &mut usize, count: usize) {
    *index += count;
    *column += count;
}

fn push_next_char(
    input: &str,
    index: &mut usize,
    line: &mut usize,
    column: &mut usize,
    output: &mut String,
) {
    if let Some(character) = input[*index..].chars().next() {
        output.push(character);
        *index += character.len_utf8();
        if character == '\n' {
            *line += 1;
            *column = 1;
        } else {
            *column += 1;
        }
    }
}

fn trim_outer_literal_whitespace(parts: &mut Vec<TemplatePart>) {
    if let Some(TemplatePart::Literal(first)) = parts.first_mut() {
        *first = first.trim_start().to_string();
    }
    if matches!(parts.first(), Some(TemplatePart::Literal(text)) if text.is_empty()) {
        parts.remove(0);
    }
    if let Some(TemplatePart::Literal(last)) = parts.last_mut() {
        *last = last.trim_end().to_string();
    }
    if matches!(parts.last(), Some(TemplatePart::Literal(text)) if text.is_empty()) {
        parts.pop();
    }
}

fn canonical_template_source(parts: &[TemplatePart]) -> String {
    let mut source = String::new();
    for part in parts {
        match part {
            TemplatePart::Literal(text) => push_escaped_delimiters(&mut source, text),
            TemplatePart::Directive(instructions) => {
                source.push_str("{{");
                push_escaped_delimiters(&mut source, instructions);
                source.push_str("}}");
            }
        }
    }
    source
}

fn push_escaped_delimiters(output: &mut String, input: &str) {
    let mut index = 0;
    while index < input.len() {
        if input[index..].starts_with("{{") {
            output.push_str("\\{{");
            index += 2;
        } else if input[index..].starts_with("}}") {
            output.push_str("\\}}");
            index += 2;
        } else if let Some(character) = input[index..].chars().next() {
            output.push(character);
            index += character.len_utf8();
        }
    }
}

fn update_hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn update_hash_template(hasher: &mut blake3::Hasher, template: &Template) {
    hasher.update(&(template.parts.len() as u64).to_le_bytes());
    for part in &template.parts {
        match part {
            TemplatePart::Literal(text) => {
                hasher.update(&[0]);
                update_hash_bytes(hasher, text.as_bytes());
            }
            TemplatePart::Directive(instructions) => {
                hasher.update(&[1]);
                update_hash_bytes(hasher, instructions.as_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use rusqlite::Connection;
    use rusqlite::params;
    use tempfile::TempDir;

    use super::*;

    type TestResult = Result<(), Box<dyn Error>>;

    #[test]
    fn parses_and_renders_mixed_static_and_generated_templates() -> TestResult {
        let question = Template::parse(
            "What is {{ a multiplication expression using two integers from 2 through 12 }}?",
        )?;
        let answer = Template::parse(
            "Result: {{ the exact integer product of the operands in the rendered question }}.",
        )?;

        assert_eq!(
            question.parts(),
            &[
                TemplatePart::Literal("What is ".to_string()),
                TemplatePart::Directive(
                    "a multiplication expression using two integers from 2 through 12".to_string(),
                ),
                TemplatePart::Literal("?".to_string()),
            ]
        );
        assert_eq!(question.directive_count(), 1);
        assert_eq!(answer.directive_count(), 1);
        assert_eq!(
            question.source(),
            "What is {{a multiplication expression using two integers from 2 through 12}}?"
        );
        assert_eq!(question.render(&["3 × 4".to_string()])?, "What is 3 × 4?");
        assert_eq!(answer.render(&["12".to_string()])?, "Result: 12.");
        Ok(())
    }

    #[test]
    fn supports_static_templates_and_escaped_delimiters() -> TestResult {
        let template = Template::parse(r"Explain \{{this notation\}} and preserve \}} too.")?;

        assert_eq!(template.directive_count(), 0);
        assert_eq!(
            template.parts(),
            &[TemplatePart::Literal(
                "Explain {{this notation}} and preserve }} too.".to_string()
            )]
        );
        assert_eq!(
            template.source(),
            r"Explain \{{this notation\}} and preserve \}} too."
        );
        assert_eq!(
            template.render(&[])?,
            "Explain {{this notation}} and preserve }} too."
        );
        Ok(())
    }

    #[test]
    fn canonical_source_round_trips_escaped_delimiters_in_directives() -> TestResult {
        let first = Template::parse(r"Before {{compare \{{x\}} with y}} after")?;
        let second = Template::parse(first.source())?;

        assert_eq!(first, second);
        assert_eq!(first.to_string(), first.source());
        assert_eq!(
            first.directives().collect::<Vec<_>>(),
            vec!["compare {{x}} with y"]
        );
        Ok(())
    }

    #[test]
    fn rejects_empty_unclosed_nested_and_unmatched_directives_with_positions() {
        let empty = Template::parse("x {{   }}").unwrap_err();
        assert_eq!((empty.line, empty.column), (1, 3));
        assert!(empty.message.contains("empty"));

        let unclosed = Template::parse("before {{oops\ncontinued").unwrap_err();
        assert_eq!((unclosed.line, unclosed.column), (1, 8));
        assert!(unclosed.message.contains("unclosed"));

        let nested = Template::parse("first\n{{outer {{inner}} }}").unwrap_err();
        assert_eq!((nested.line, nested.column), (2, 9));
        assert!(nested.message.contains("nested"));

        let unmatched = Template::parse("first\nraw }}").unwrap_err();
        assert_eq!((unmatched.line, unmatched.column), (2, 5));
        assert!(unmatched.message.contains("no matching"));
    }

    #[test]
    fn render_requires_exact_replacement_count() -> TestResult {
        let template = Template::parse("{{first}} + {{second}}")?;

        assert_eq!(
            template.render(&["one".to_string()]).unwrap_err(),
            TemplateRenderError {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            template
                .render(&["one".to_string(), "two".to_string(), "three".to_string()])
                .unwrap_err(),
            TemplateRenderError {
                expected: 2,
                actual: 3,
            }
        );
        Ok(())
    }

    #[test]
    fn parses_frontmatter_goal_and_multiline_fields() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Arithmetic.md");
        fs::write(
            &path,
            r#"+++
name = "Mental arithmetic"
source = "https://example.test/multiplication"
+++

G: Fluently multiply positive integers.
   {{This remains plain hidden goal text.}}
Q: What is {{an expression using two integers from 2 through 12}}?
   Give only the result.
A:
{{the exact integer product}}
"#,
        )?;

        let specs = parse_path(&path)?;
        assert_eq!(specs.len(), 1);
        let spec = &specs[0];
        assert_eq!(spec.deck_name, "Mental arithmetic");
        assert_eq!(
            spec.source.as_deref(),
            Some("https://example.test/multiplication")
        );
        assert_eq!(spec.path, path.canonicalize()?);
        assert_eq!(spec.range, (6, 11));
        assert_eq!(
            spec.goal.as_deref(),
            Some(
                "Fluently multiply positive integers.\n   {{This remains plain hidden goal text.}}"
            )
        );
        assert_eq!(spec.question.directive_count(), 1);
        assert_eq!(spec.answer.directive_count(), 1);
        assert_eq!(
            spec.question.render(&["3 × 4".to_string()])?,
            "What is 3 × 4?\n   Give only the result."
        );
        Ok(())
    }

    #[test]
    fn bundled_probability_example_preserves_latex_around_generation() -> TestResult {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("example")
            .join("Conditional-Probability.md");
        let specs = parse_path(&path)?;
        assert_eq!(specs.len(), 1);

        let spec = &specs[0];
        assert_eq!(spec.question.directive_count(), 4);
        assert_eq!(spec.answer.directive_count(), 1);
        assert!(
            spec.goal
                .as_deref()
                .is_some_and(|goal| goal.contains(r"$P(A \mid B)=\frac{n(A \cap B)}{n(B)}$"))
        );

        let rendered = spec.question.render(&[
            "4".to_string(),
            "7".to_string(),
            "2".to_string(),
            "9".to_string(),
        ])?;
        assert!(rendered.contains(r"| | $B$ | $\neg B$ |"));
        assert!(rendered.contains("$$\nP(A \\mid B)\n$$"));
        assert!(!rendered.contains("{{"));
        Ok(())
    }

    #[test]
    fn trims_and_accepts_hashcards_source_schemes() -> TestResult {
        let directory = TempDir::new()?;
        let cases = [
            (
                "Https.md",
                "  https://example.test/reference  ",
                "https://example.test/reference",
            ),
            (
                "Http.md",
                " http://example.test/reference ",
                "http://example.test/reference",
            ),
            (
                "Obsidian.md",
                "  obsidian://open?vault=Notes&file=Reference  ",
                "obsidian://open?vault=Notes&file=Reference",
            ),
        ];

        for (file_name, authored_source, expected_source) in cases {
            let path = directory.path().join(file_name);
            fs::write(
                &path,
                format!("+++\nsource = {authored_source:?}\n+++\nQ: A question?\nA: An answer.\n"),
            )?;

            let spec = parse_path(&path)?.remove(0);
            assert_eq!(spec.source.as_deref(), Some(expected_source));
        }
        Ok(())
    }

    #[test]
    fn rejects_empty_and_unsafe_sources_at_the_frontmatter_key() -> TestResult {
        let directory = TempDir::new()?;
        let cases = [
            ("Empty.md", "   ", "frontmatter source must not be empty"),
            (
                "Unsafe.md",
                "javascript:alert(1)",
                "frontmatter source must use http://, https://, or obsidian://",
            ),
        ];

        for (file_name, source, expected_message) in cases {
            let path = directory.path().join(file_name);
            fs::write(
                &path,
                format!(
                    "+++\nname = \"Source validation\"\n  source = {source:?}\n+++\nQ: A question?\nA: An answer.\n"
                ),
            )?;

            let error = parse_path(&path).unwrap_err();
            assert_eq!(error.message, expected_message);
            assert_eq!(error.path, path.canonicalize()?);
            assert_eq!((error.line, error.column), (3, 3));
            assert!(
                error
                    .to_string()
                    .contains(&error.path.display().to_string())
            );
        }
        Ok(())
    }

    #[test]
    fn permits_zero_directive_typed_answer_drills() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Static.md");
        fs::write(&path, "Q: What's 3 × 4?\nA: 12.\n")?;

        let specs = parse_path(&path)?;
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].question.directive_count(), 0);
        assert_eq!(specs[0].answer.directive_count(), 0);
        assert_eq!(specs[0].question.render(&[])?, "What's 3 × 4?");
        assert_eq!(specs[0].answer.render(&[])?, "12.");
        Ok(())
    }

    #[test]
    fn accepts_separators_and_implicit_next_drill_boundaries() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Mixed.md");
        fs::write(
            &path,
            "Q: Static one?\nA: One.\n---\nG: Hidden two\nQ: {{Question two}}?\nA: {{Answer two}}.\nQ: Static three?\nA: Three.\n",
        )?;

        let specs = parse_path(&path)?;
        assert_eq!(specs.len(), 3);
        let ranges: Vec<(usize, usize)> = specs.iter().map(|spec| spec.range).collect();
        assert!(ranges.contains(&(1, 2)));
        assert!(ranges.contains(&(4, 6)));
        assert!(ranges.contains(&(7, 8)));
        Ok(())
    }

    #[test]
    fn recursively_parses_markdown_and_skips_appledouble_files() -> TestResult {
        let directory = TempDir::new()?;
        let nested = directory.path().join("nested");
        fs::create_dir_all(&nested)?;
        fs::write(directory.path().join("One.md"), "Q: One?\nA: One.\n")?;
        fs::write(nested.join("Two.MD"), "Q: Two?\nA: Two.\n")?;
        fs::write(
            directory.path().join("._Ghost.md"),
            "this is invalid AppleDouble metadata",
        )?;
        fs::write(directory.path().join("ignored.txt"), "not Markdown")?;

        let specs = parse_path(directory.path())?;
        assert_eq!(specs.len(), 2);
        assert!(specs.iter().any(|spec| spec.deck_name == "One"));
        assert!(specs.iter().any(|spec| spec.deck_name == "Two"));
        Ok(())
    }

    #[test]
    fn recursively_skips_plaintext_generated_archives() -> TestResult {
        let directory = TempDir::new()?;
        fs::write(
            directory.path().join("Authored.md"),
            "Q: Authored?\nA: Yes.\n",
        )?;
        fs::write(
            directory.path().join("Archived.md"),
            r#"+++
hashdrills_archive_kind = "generated_drill"
hashdrills_archive_format_version = 1
name = "Archived evidence"
generated_at = "2026-07-31T12:00:00.000"
unexpected_future_archive_field = "ignored with the archive"
+++

Q: Frozen generated question?
A: Frozen generated criteria.
"#,
        )?;

        let specs = parse_path(directory.path())?;
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].deck_name, "Authored");
        Ok(())
    }

    #[test]
    fn rejects_unsupported_archive_format_versions_at_the_marker() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Future-archive.md");
        fs::write(
            &path,
            r#"+++
hashdrills_archive_kind = "generated_drill"
  hashdrills_archive_format_version = 2
name = "Future evidence"
+++

Q: Must not become schedulable?
A: Yes.
"#,
        )?;

        let error = parse_path(&path).unwrap_err();
        assert_eq!(
            error.message,
            "unsupported archive format version 2; supported version is 1"
        );
        assert_eq!((error.line, error.column), (3, 3));
        Ok(())
    }

    #[test]
    fn malformed_archive_markers_do_not_bypass_frontmatter_validation() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Malformed-archive.md");
        fs::write(
            &path,
            r#"+++
hashdrills_archive_kind = "generated_drill"
hashdrills_archive_format_version = "1"
unexpected_future_archive_field = "must not be ignored"
+++

Q: Must not become schedulable?
A: Yes.
"#,
        )?;

        let error = parse_path(&path).unwrap_err();
        assert!(
            error
                .message
                .starts_with("failed to parse TOML frontmatter:")
        );
        assert!(error.message.contains("invalid type"));
        assert_eq!((error.line, error.column), (2, 1));
        Ok(())
    }

    #[test]
    fn archive_marker_lookalikes_remain_strict_unknown_metadata() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Marker-lookalike.md");
        fs::write(
            &path,
            r#"+++
hashdrills_archive_format_version_extra = 1
+++

Q: Must remain authored input?
A: Yes.
"#,
        )?;

        let error = parse_path(&path).unwrap_err();
        assert!(
            error
                .message
                .starts_with("failed to parse TOML frontmatter:")
        );
        assert!(
            error
                .message
                .contains("unknown field `hashdrills_archive_format_version_extra`")
        );
        assert_eq!((error.line, error.column), (2, 1));
        Ok(())
    }

    #[test]
    fn partial_archive_markers_are_errors_instead_of_skip_signals() -> TestResult {
        let directory = TempDir::new()?;
        for (filename, marker, missing_key) in [
            (
                "Kind-only.md",
                "hashdrills_archive_kind = \"generated_drill\"",
                "hashdrills_archive_format_version",
            ),
            (
                "Version-only.md",
                "hashdrills_archive_format_version = 1",
                "hashdrills_archive_kind",
            ),
        ] {
            let path = directory.path().join(filename);
            fs::write(
                &path,
                format!("+++\n{marker}\n+++\n\nQ: Authored?\nA: Yes.\n"),
            )?;

            let error = parse_path(&path).unwrap_err();
            assert!(
                error
                    .message
                    .starts_with("incomplete Hashdrills archive marker:")
            );
            assert!(error.message.contains(missing_key));
            assert_eq!((error.line, error.column), (2, 1));
        }
        Ok(())
    }

    #[test]
    fn unsupported_archive_kinds_are_errors_instead_of_skip_signals() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Wrong-kind.md");
        fs::write(
            &path,
            r#"+++
  hashdrills_archive_kind = "unrelated_document"
hashdrills_archive_format_version = 1
+++

Q: Authored?
A: Yes.
"#,
        )?;

        let error = parse_path(&path).unwrap_err();
        assert_eq!(
            error.message,
            "unsupported Hashdrills archive kind \"unrelated_document\"; supported kind is \"generated_drill\""
        );
        assert_eq!((error.line, error.column), (2, 3));
        Ok(())
    }

    #[test]
    fn old_generic_archive_version_key_cannot_suppress_authored_content() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Authored.md");
        fs::write(
            &path,
            r#"+++
archive_format_version = 1
+++

Q: This remains authored input?
A: Yes.
"#,
        )?;

        let error = parse_path(&path).unwrap_err();
        assert!(
            error
                .message
                .contains("unknown field `archive_format_version`")
        );
        Ok(())
    }

    #[test]
    fn recursively_skips_supported_session_transcripts() -> TestResult {
        let directory = TempDir::new()?;
        fs::write(
            directory.path().join("Authored.md"),
            "Q: Authored?\nA: Yes.\n",
        )?;
        fs::write(
            directory.path().join("Session.md"),
            r#"+++
hashdrills_session_log_kind = "session_transcript"
hashdrills_session_log_format_version = 1
future_transcript_metadata = "ignored with the transcript"
+++

# Hashdrills session transcript

```text
Q: This fenced evidence must never become a drill.
A: Nor should this.
```
"#,
        )?;

        let specs = parse_path(directory.path())?;
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].deck_name, "Authored");
        Ok(())
    }

    #[test]
    fn partial_and_unsupported_session_log_markers_fail_closed() -> TestResult {
        let directory = TempDir::new()?;
        let cases = [
            (
                "Kind-only.md",
                "hashdrills_session_log_kind = \"session_transcript\"",
                "incomplete Hashdrills session-log marker:",
            ),
            (
                "Version-only.md",
                "hashdrills_session_log_format_version = 1",
                "incomplete Hashdrills session-log marker:",
            ),
            (
                "Future.md",
                "hashdrills_session_log_kind = \"session_transcript\"\nhashdrills_session_log_format_version = 2",
                "unsupported session transcript format version 2; supported version is 1",
            ),
            (
                "Wrong-kind.md",
                "hashdrills_session_log_kind = \"unrelated_document\"\nhashdrills_session_log_format_version = 1",
                "unsupported Hashdrills session-log kind \"unrelated_document\"; supported kind is \"session_transcript\"",
            ),
        ];
        for (filename, marker, expected) in cases {
            let path = directory.path().join(filename);
            fs::write(
                &path,
                format!("+++\n{marker}\n+++\n\nQ: Authored?\nA: Yes.\n"),
            )?;
            let error = parse_path(&path).unwrap_err();
            assert!(
                error.message.starts_with(expected),
                "unexpected diagnostic: {}",
                error.message
            );
        }
        Ok(())
    }

    #[test]
    fn mixed_generated_output_marker_families_fail_closed() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Mixed-markers.md");
        fs::write(
            &path,
            r#"+++
hashdrills_archive_kind = "generated_drill"
hashdrills_archive_format_version = 1
hashdrills_session_log_kind = "session_transcript"
hashdrills_session_log_format_version = 1
+++

Q: Authored?
A: Yes.
"#,
        )?;

        let error = parse_path(&path).unwrap_err();
        assert_eq!(
            error.message,
            "a file may contain only one Hashdrills generated-output marker family"
        );
        Ok(())
    }

    #[test]
    fn generic_and_lookalike_session_keys_remain_strict_metadata_errors() -> TestResult {
        let directory = TempDir::new()?;
        for (filename, key) in [
            ("Generic.md", "session_log_format_version"),
            (
                "Lookalike.md",
                "hashdrills_session_log_format_version_extra",
            ),
        ] {
            let path = directory.path().join(filename);
            fs::write(
                &path,
                format!("+++\n{key} = 1\n+++\n\nQ: Authored?\nA: Yes.\n"),
            )?;
            let error = parse_path(&path).unwrap_err();
            assert!(error.message.contains(&format!("unknown field `{key}`")));
        }
        Ok(())
    }

    #[test]
    fn hash_is_stable_across_path_deck_name_and_source_frontmatter() -> TestResult {
        let first_directory = TempDir::new()?;
        let second_directory = TempDir::new()?;
        let first_path = first_directory.path().join("First.md");
        let second_path = second_directory.path().join("Second.md");
        fs::write(
            &first_path,
            r#"+++
name = "First deck"
source = "https://example.test/one"
+++
G: Multiply bounded integers.
Q: What is {{an integer multiplication expression}}?
A: {{the exact product}}.
"#,
        )?;
        fs::write(
            &second_path,
            r#"+++
name = "A renamed deck"
source = "obsidian://open?vault=Notes&file=AnotherSource"
+++
G: Multiply bounded integers.
Q: What is {{ an integer multiplication expression }}?
A: {{ the exact product }}.
"#,
        )?;

        let first = parse_path(&first_path)?.remove(0);
        let second = parse_path(&second_path)?.remove(0);
        assert_ne!(first.path, second.path);
        assert_ne!(first.deck_name, second.deck_name);
        assert_ne!(first.source, second.source);
        assert_eq!(first.question, second.question);
        assert_eq!(first.hash(), second.hash());
        Ok(())
    }

    #[test]
    fn content_changes_change_the_hash_without_ambiguous_field_concatenation() -> TestResult {
        let path = PathBuf::from("fixture.md");
        let first = DrillSpec::new(
            "Deck",
            None,
            path.clone(),
            (1, 2),
            None,
            Template::parse("a")?,
            Template::parse("bc")?,
        );
        let second = DrillSpec::new(
            "Deck",
            None,
            path,
            (1, 2),
            None,
            Template::parse("ab")?,
            Template::parse("c")?,
        );

        assert_ne!(first.hash(), second.hash());
        assert_eq!(first.hash().to_hex().len(), 64);
        assert_eq!(
            first.hash().to_hex(),
            "79361c227226ee2d195d2128c03063bb22f0c7cf1aac9ee566db996fe9a4af0a"
        );
        Ok(())
    }

    #[test]
    fn duplicate_hashes_are_reported_instead_of_silently_removed() -> TestResult {
        let directory = TempDir::new()?;
        fs::write(directory.path().join("One.md"), "Q: Same?\nA: Same.\n")?;
        fs::write(directory.path().join("Two.md"), "Q: Same?\nA: Same.\n")?;

        let error = parse_path(directory.path()).unwrap_err();
        assert!(error.message.contains("duplicate drill specification"));
        assert!(error.message.contains("One.md"));
        assert_eq!(error.line, 1);
        assert_eq!(error.column, 1);
        Ok(())
    }

    #[test]
    fn file_errors_preserve_directive_line_and_unicode_column() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Broken.md");
        fs::write(
            &path,
            "G: A plain goal with {{braces}}.\nQ: First line\né {{outer {{nested}} }}\nA: target\n",
        )?;

        let error = parse_path(&path).unwrap_err();
        assert_eq!(error.path, path.canonicalize()?);
        assert_eq!((error.line, error.column), (3, 11));
        assert!(error.message.contains("nested"));
        Ok(())
    }

    #[test]
    fn tags_must_begin_at_column_zero() -> TestResult {
        let directory = TempDir::new()?;
        let path = directory.path().join("Indented.md");
        fs::write(&path, "  Q: Not a tag\nA: Not reached\n")?;

        let error = parse_path(&path).unwrap_err();
        assert_eq!((error.line, error.column), (1, 1));
        assert!(error.message.contains("column 0"));
        Ok(())
    }

    #[test]
    fn spec_hash_round_trips_through_serde_and_sqlite() -> TestResult {
        let spec = DrillSpec::new(
            "Deck",
            None,
            PathBuf::from("fixture.md"),
            (1, 3),
            Some("A goal".to_string()),
            Template::parse("Question {{variation}}?")?,
            Template::parse("{{target}}")?,
        );
        let hash = spec.hash();

        let json = serde_json::to_string(&hash)?;
        assert_eq!(serde_json::from_str::<SpecHash>(&json)?, hash);
        assert_eq!(SpecHash::from_hex(&hash.to_hex())?, hash);

        let connection = Connection::open_in_memory()?;
        connection.execute("CREATE TABLE specs (hash TEXT NOT NULL)", [])?;
        connection.execute("INSERT INTO specs (hash) VALUES (?1)", params![hash])?;
        let loaded: SpecHash =
            connection.query_row("SELECT hash FROM specs", [], |row| row.get(0))?;
        assert_eq!(loaded, hash);
        Ok(())
    }

    #[test]
    fn missing_and_empty_fields_are_clear_errors() -> TestResult {
        let directory = TempDir::new()?;
        let no_answer = directory.path().join("NoAnswer.md");
        fs::write(&no_answer, "Q: Question only\n")?;
        assert!(
            parse_path(&no_answer)
                .unwrap_err()
                .message
                .contains("before Q: had an A:")
        );

        let empty_answer = directory.path().join("EmptyAnswer.md");
        fs::write(&empty_answer, "Q: Question\nA:   \n")?;
        assert!(
            parse_path(&empty_answer)
                .unwrap_err()
                .message
                .contains("answer cannot be empty")
        );

        let empty_goal = directory.path().join("EmptyGoal.md");
        fs::write(&empty_goal, "G:  \nQ: Question\nA: Answer\n")?;
        assert!(
            parse_path(&empty_goal)
                .unwrap_err()
                .message
                .contains("goal cannot be empty")
        );
        Ok(())
    }
}
