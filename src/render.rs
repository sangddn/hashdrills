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

//! Safe rich-text rendering for generated drills.
//!
//! Drill text is partly model-generated and therefore treated as untrusted.
//! Markdown is supported, but authored HTML is emitted as visible text rather
//! than passed through to the browser. Local media references are resolved
//! within the collection and exposed through the `/file/*path` route.

use std::fmt::Display;
use std::fmt::Formatter;
use std::fs;
use std::ops::Range;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use pulldown_cmark::CowStr;
use pulldown_cmark::Event;
use pulldown_cmark::Options;
use pulldown_cmark::Parser;
use pulldown_cmark::Tag;
use pulldown_cmark::TagEnd;
use pulldown_cmark::html::push_html;

/// The browser element used for a supported local media file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Audio,
    Video,
}

/// A validated collection file ready to be served by the web layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaFile {
    pub path: PathBuf,
    pub content_type: &'static str,
    pub kind: MediaKind,
    pub(crate) identity: MediaIdentity,
}

/// Stable identity captured while a media request is validated.
///
/// The web layer compares this with the opened handle before streaming any
/// bytes. This closes the gap where a collection file (or one of its parent
/// directories) is replaced after path validation but before `open(2)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MediaIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(not(unix))]
    length: u64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
    #[cfg(not(unix))]
    created: Option<std::time::SystemTime>,
}

impl MediaIdentity {
    pub(crate) fn capture(metadata: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            Self {
                device: metadata.dev(),
                inode: metadata.ino(),
            }
        }
        #[cfg(not(unix))]
        {
            Self {
                length: metadata.len(),
                modified: metadata.modified().ok(),
                created: metadata.created().ok(),
            }
        }
    }

    pub(crate) fn matches(&self, metadata: &fs::Metadata) -> bool {
        self == &Self::capture(metadata)
    }
}

/// A local Markdown media destination rewritten for the web application.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedMedia {
    pub url: String,
    pub kind: MediaKind,
}

/// A rejected or unusable local media path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaError {
    InvalidRoot,
    Empty,
    External,
    Absolute,
    ParentTraversal,
    OutsideCollection,
    Symlink,
    NotFound,
    NotFile,
    UnsupportedType,
    NonUtf8Path,
}

impl Display for MediaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidRoot => "the collection root is not a directory",
            Self::Empty => "the media path is empty",
            Self::External => "external media URLs are not allowed",
            Self::Absolute => "absolute media paths are not allowed",
            Self::ParentTraversal => "the request contains a parent path component",
            Self::OutsideCollection => "the media path is outside the collection",
            Self::Symlink => "symbolic links are not served",
            Self::NotFound => "the media file does not exist",
            Self::NotFile => "the media path is not a file",
            Self::UnsupportedType => "the media type is not supported",
            Self::NonUtf8Path => "the media path is not valid UTF-8",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for MediaError {}

/// Render block Markdown for one drill field.
///
/// This function deliberately cannot fail. An invalid local-media reference
/// becomes a visible, escaped fallback while the rest of the Markdown remains
/// usable.
pub fn markdown_to_html(markdown: &str, collection_root: &Path, spec_path: &Path) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    options.insert(Options::ENABLE_SMART_PUNCTUATION);
    options.insert(Options::ENABLE_MATH);
    options.insert(Options::ENABLE_GFM);

    let rewritten = rewrite_latex_delimiters(markdown);
    let parser = Parser::new_ext(&rewritten, options);
    let mut rendered_events = Vec::new();
    let mut image: Option<ImageCapture> = None;
    let mut link_stack = Vec::new();

    for event in parser {
        if image.is_some() {
            if matches!(&event, Event::End(TagEnd::Image)) {
                if let Some(capture) = image.take() {
                    rendered_events.push(Event::Html(CowStr::Boxed(
                        render_media(capture, collection_root, spec_path).into_boxed_str(),
                    )));
                }
            } else if let Some(capture) = image.as_mut() {
                capture.push_alt(event);
            }
            continue;
        }

        match event {
            Event::Start(Tag::Image {
                title, dest_url, ..
            }) => {
                image = Some(ImageCapture {
                    destination: dest_url.into_string(),
                    title: title.into_string(),
                    alt: String::new(),
                });
            }
            Event::Start(Tag::Link {
                title, dest_url, ..
            }) => {
                if let Some(opening) = render_link_open(&dest_url, &title) {
                    link_stack.push(true);
                    rendered_events.push(Event::Html(CowStr::Boxed(opening.into_boxed_str())));
                } else {
                    // Preserve the link label, but omit an unsafe anchor.
                    link_stack.push(false);
                }
            }
            Event::End(TagEnd::Link) => {
                if link_stack.pop().unwrap_or(false) {
                    rendered_events.push(Event::Html(CowStr::Borrowed("</a>")));
                }
            }
            // Model output is untrusted. Showing raw HTML as text is both safer
            // and less surprising than silently deleting it.
            Event::Html(raw) | Event::InlineHtml(raw) => {
                rendered_events.push(Event::Text(raw));
            }
            other => rendered_events.push(other),
        }
    }

    // Pulldown-cmark normally balances image events. Retain a defensive
    // fallback in case that invariant ever changes for malformed input.
    if let Some(capture) = image {
        rendered_events.push(Event::Html(CowStr::Boxed(
            media_fallback(&capture, MediaError::NotFile).into_boxed_str(),
        )));
    }

    let mut output = String::new();
    push_html(&mut output, rendered_events.into_iter());
    output
}

/// Render Markdown while removing only a single surrounding paragraph.
pub fn markdown_to_html_inline(markdown: &str, collection_root: &Path, spec_path: &Path) -> String {
    let rendered = markdown_to_html(markdown, collection_root, spec_path);
    rendered
        .strip_prefix("<p>")
        .and_then(|body| body.strip_suffix("</p>\n"))
        .unwrap_or(&rendered)
        .to_string()
}

/// Resolve a Markdown image destination according to Hashcards conventions.
///
/// `@/foo.png` is collection-root-relative. Every other non-absolute path is
/// relative to the Markdown file containing the drill. Parent components are
/// accepted for file-relative paths only when the canonical target remains
/// inside the collection.
pub fn resolve_media_reference(
    collection_root: &Path,
    spec_path: &Path,
    destination: &str,
) -> Result<ResolvedMedia, MediaError> {
    match resolve_media_reference_inner(collection_root, spec_path, destination) {
        Ok(media) => Ok(media),
        Err(MediaError::NotFound) => {
            let decoded = percent_decode(destination).ok_or(MediaError::NotFound)?;
            if decoded == destination {
                Err(MediaError::NotFound)
            } else {
                resolve_media_reference_inner(collection_root, spec_path, &decoded)
            }
        }
        Err(error) => Err(error),
    }
}

/// Validate an untrusted wildcard path received by the `/file/*path` route.
///
/// Absolute paths, parent components, every symbolic-link component, paths
/// outside the canonical root, directories, missing files, and unsupported
/// extensions are rejected.
pub fn validate_media_request(
    collection_root: &Path,
    request_path: &str,
) -> Result<MediaFile, MediaError> {
    match validate_media_request_inner(collection_root, request_path) {
        Ok(file) => Ok(file),
        Err(MediaError::NotFound) => {
            let decoded = percent_decode(request_path).ok_or(MediaError::NotFound)?;
            if decoded == request_path {
                Err(MediaError::NotFound)
            } else {
                validate_media_request_inner(collection_root, &decoded)
            }
        }
        Err(error) => Err(error),
    }
}

fn resolve_media_reference_inner(
    collection_root: &Path,
    spec_path: &Path,
    destination: &str,
) -> Result<ResolvedMedia, MediaError> {
    let destination = destination.trim();
    if destination.is_empty() {
        return Err(MediaError::Empty);
    }
    if looks_like_external_url(destination) {
        return Err(MediaError::External);
    }

    let root = canonical_collection_root(collection_root)?;
    let (candidate, root_relative) = if let Some(path) = destination.strip_prefix("@/") {
        let relative = Path::new(path);
        reject_absolute(relative)?;
        if relative
            .components()
            .any(|component| component == Component::ParentDir)
        {
            return Err(MediaError::ParentTraversal);
        }
        (root.join(relative), true)
    } else {
        let relative = Path::new(destination);
        reject_absolute(relative)?;
        let spec = if spec_path.is_absolute() {
            spec_path.to_path_buf()
        } else {
            root.join(spec_path)
        };
        let spec = spec.canonicalize().map_err(|_| MediaError::NotFound)?;
        if !spec.starts_with(&root) {
            return Err(MediaError::OutsideCollection);
        }
        let directory = spec.parent().ok_or(MediaError::NotFound)?;
        (directory.join(relative), false)
    };

    if !candidate.exists() {
        return Err(MediaError::NotFound);
    }
    let candidate = candidate.canonicalize().map_err(|_| MediaError::NotFound)?;
    if !candidate.starts_with(&root) {
        return Err(MediaError::OutsideCollection);
    }
    if !candidate.is_file() {
        return Err(MediaError::NotFile);
    }

    // Collection-root references are intentionally stricter: unlike a deck
    // relative path they never need `..` to reach a sibling directory.
    if root_relative && destination[2..].split('/').any(|part| part == "..") {
        return Err(MediaError::ParentTraversal);
    }

    let relative = candidate
        .strip_prefix(&root)
        .map_err(|_| MediaError::OutsideCollection)?;
    let (kind, _) = media_type(relative)?;
    let encoded = percent_encode_path(relative)?;
    Ok(ResolvedMedia {
        url: format!("/file/{encoded}"),
        kind,
    })
}

fn validate_media_request_inner(
    collection_root: &Path,
    request_path: &str,
) -> Result<MediaFile, MediaError> {
    if request_path.is_empty() {
        return Err(MediaError::Empty);
    }
    let relative = Path::new(request_path);
    reject_absolute(relative)?;
    for component in relative.components() {
        match component {
            Component::ParentDir => return Err(MediaError::ParentTraversal),
            Component::RootDir | Component::Prefix(_) => return Err(MediaError::Absolute),
            Component::CurDir | Component::Normal(_) => {}
        }
    }

    let root = canonical_collection_root(collection_root)?;
    reject_symlink_components(&root, relative)?;
    let candidate = root.join(relative);
    if !candidate.exists() {
        return Err(MediaError::NotFound);
    }
    if !candidate.is_file() {
        return Err(MediaError::NotFile);
    }
    let candidate = candidate.canonicalize().map_err(|_| MediaError::NotFound)?;
    if !candidate.starts_with(&root) {
        return Err(MediaError::OutsideCollection);
    }
    let (kind, content_type) = media_type(&candidate)?;
    let metadata = fs::symlink_metadata(&candidate).map_err(|_| MediaError::NotFound)?;
    if metadata.file_type().is_symlink() {
        return Err(MediaError::Symlink);
    }
    if !metadata.is_file() {
        return Err(MediaError::NotFile);
    }
    let identity = MediaIdentity::capture(&metadata);

    // Pin the identity to the canonical target, not merely to the path text.
    // A parent swapped for a symlink between the first canonicalization and
    // the metadata read changes either the canonical path or the identity.
    let rechecked = candidate.canonicalize().map_err(|_| MediaError::NotFound)?;
    if rechecked != candidate || !rechecked.starts_with(&root) {
        return Err(MediaError::Symlink);
    }
    let rechecked_metadata = fs::symlink_metadata(&candidate).map_err(|_| MediaError::NotFound)?;
    if rechecked_metadata.file_type().is_symlink()
        || !rechecked_metadata.is_file()
        || !identity.matches(&rechecked_metadata)
    {
        return Err(MediaError::Symlink);
    }
    Ok(MediaFile {
        path: candidate,
        content_type,
        kind,
        identity,
    })
}

fn canonical_collection_root(collection_root: &Path) -> Result<PathBuf, MediaError> {
    let root = collection_root
        .canonicalize()
        .map_err(|_| MediaError::InvalidRoot)?;
    if !root.is_dir() {
        return Err(MediaError::InvalidRoot);
    }
    Ok(root)
}

fn reject_absolute(path: &Path) -> Result<(), MediaError> {
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::RootDir | Component::Prefix(_)))
    {
        Err(MediaError::Absolute)
    } else {
        Ok(())
    }
}

fn reject_symlink_components(root: &Path, relative: &Path) -> Result<(), MediaError> {
    let mut candidate = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::CurDir => continue,
            Component::Normal(name) => candidate.push(name),
            Component::ParentDir => return Err(MediaError::ParentTraversal),
            Component::RootDir | Component::Prefix(_) => return Err(MediaError::Absolute),
        }
        match fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(MediaError::Symlink);
            }
            Ok(_) => {}
            Err(_) => return Err(MediaError::NotFound),
        }
    }
    Ok(())
}

fn media_type(path: &Path) -> Result<(MediaKind, &'static str), MediaError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .ok_or(MediaError::UnsupportedType)?
        .to_ascii_lowercase();
    let result = match extension.as_str() {
        "png" => (MediaKind::Image, "image/png"),
        "jpg" | "jpeg" => (MediaKind::Image, "image/jpeg"),
        "gif" => (MediaKind::Image, "image/gif"),
        "svg" => (MediaKind::Image, "image/svg+xml"),
        "webp" => (MediaKind::Image, "image/webp"),
        "bmp" => (MediaKind::Image, "image/bmp"),
        "avif" => (MediaKind::Image, "image/avif"),
        "mp3" => (MediaKind::Audio, "audio/mpeg"),
        "wav" => (MediaKind::Audio, "audio/wav"),
        "ogg" => (MediaKind::Audio, "audio/ogg"),
        "m4a" => (MediaKind::Audio, "audio/mp4"),
        "flac" => (MediaKind::Audio, "audio/flac"),
        "mp4" => (MediaKind::Video, "video/mp4"),
        "webm" => (MediaKind::Video, "video/webm"),
        "mov" => (MediaKind::Video, "video/quicktime"),
        _ => return Err(MediaError::UnsupportedType),
    };
    Ok(result)
}

struct ImageCapture {
    destination: String,
    title: String,
    alt: String,
}

impl ImageCapture {
    fn push_alt(&mut self, event: Event<'_>) {
        match event {
            Event::Text(text)
            | Event::Code(text)
            | Event::InlineMath(text)
            | Event::DisplayMath(text)
            | Event::Html(text)
            | Event::InlineHtml(text)
            | Event::FootnoteReference(text) => self.alt.push_str(&text),
            Event::SoftBreak | Event::HardBreak => self.alt.push(' '),
            Event::Rule => self.alt.push_str(" — "),
            Event::TaskListMarker(checked) => {
                self.alt.push_str(if checked { "[x] " } else { "[ ] " });
            }
            Event::Start(_) | Event::End(_) => {}
        }
    }
}

fn render_media(capture: ImageCapture, collection_root: &Path, spec_path: &Path) -> String {
    match resolve_media_reference(collection_root, spec_path, &capture.destination) {
        Ok(media) => {
            let source = escape_html_attribute(&media.url);
            let alt = escape_html_attribute(&capture.alt);
            let title = if capture.title.is_empty() {
                String::new()
            } else {
                format!(" title=\"{}\"", escape_html_attribute(capture.title.trim()))
            };
            match media.kind {
                MediaKind::Image => format!(
                    "<img src=\"{source}\" alt=\"{alt}\"{title} loading=\"lazy\" decoding=\"async\" />"
                ),
                MediaKind::Audio => format!(
                    "<audio controls preload=\"metadata\" src=\"{source}\"{title}>{}</audio>",
                    escape_html_text(&capture.alt)
                ),
                MediaKind::Video => format!(
                    "<video controls preload=\"metadata\" src=\"{source}\"{title}>{}</video>",
                    escape_html_text(&capture.alt)
                ),
            }
        }
        Err(error) => media_fallback(&capture, error),
    }
}

fn media_fallback(capture: &ImageCapture, error: MediaError) -> String {
    let label = if capture.alt.trim().is_empty() {
        capture.destination.trim()
    } else {
        capture.alt.trim()
    };
    format!(
        "<span class=\"media-fallback\" role=\"note\" title=\"{}\">[Media unavailable: {}]</span>",
        escape_html_attribute(&error.to_string()),
        escape_html_text(label)
    )
}

fn render_link_open(destination: &str, title: &str) -> Option<String> {
    let destination = destination.trim();
    let external = safe_link_kind(destination)?;
    let mut opening = format!("<a href=\"{}\"", escape_html_attribute(destination));
    if !title.is_empty() {
        opening.push_str(&format!(
            " title=\"{}\"",
            escape_html_attribute(title.trim())
        ));
    }
    if external {
        opening.push_str(" target=\"_blank\" rel=\"noopener noreferrer\"");
    }
    opening.push('>');
    Some(opening)
}

/// Return whether a safe link is external. Unsafe schemes return `None`.
fn safe_link_kind(destination: &str) -> Option<bool> {
    if destination.chars().any(|character| character.is_control()) || destination.contains('\\') {
        return None;
    }

    let lowercase = destination.to_ascii_lowercase();
    if lowercase.starts_with("http://")
        || lowercase.starts_with("https://")
        || lowercase.starts_with("mailto:")
        || lowercase.starts_with("obsidian://")
    {
        return Some(true);
    }
    if destination.starts_with("//") {
        return None;
    }

    // A colon before the first path/query/fragment separator denotes an
    // unapproved URI scheme such as javascript:, data:, or file:.
    let first_separator = destination
        .find(['/', '?', '#'])
        .unwrap_or(destination.len());
    if destination[..first_separator].contains(':') {
        return None;
    }
    Some(false)
}

fn looks_like_external_url(path: &str) -> bool {
    if path.starts_with("//") || path.contains("://") {
        return true;
    }
    let first_separator = path.find(['/', '?', '#']).unwrap_or(path.len());
    path[..first_separator].contains(':')
}

fn escape_html_text(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn escape_html_attribute(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn percent_encode_path(path: &Path) -> Result<String, MediaError> {
    let mut encoded = String::new();
    for component in path.components() {
        let Component::Normal(segment) = component else {
            return Err(MediaError::OutsideCollection);
        };
        let segment = segment.to_str().ok_or(MediaError::NonUtf8Path)?;
        if !encoded.is_empty() {
            encoded.push('/');
        }
        for byte in segment.as_bytes() {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                encoded.push(char::from(*byte));
            } else {
                encoded.push('%');
                encoded.push(hex_digit(byte >> 4));
                encoded.push(hex_digit(byte & 0x0f));
            }
        }
    }
    Ok(encoded)
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        _ => char::from(b'A' + value - 10),
    }
}

fn percent_decode(input: &str) -> Option<String> {
    if !input.as_bytes().contains(&b'%') {
        return Some(input.to_string());
    }
    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return None;
            }
            let high = from_hex_digit(bytes[index + 1])?;
            let low = from_hex_digit(bytes[index + 2])?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

fn from_hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

/// Convert Hashcards' `\(...\)` and `\[...\]` delimiters to the dollar
/// notation recognized by pulldown-cmark. Code spans and fenced code blocks
/// remain byte-for-byte unchanged, as do links, image destinations, and raw
/// HTML that will later be escaped.
fn rewrite_latex_delimiters(markdown: &str) -> String {
    let protected = protected_markdown_ranges(markdown);
    let bytes = markdown.as_bytes();
    let mut output = String::with_capacity(bytes.len());
    let mut copied = 0;
    let mut index = 0;

    while index + 1 < bytes.len() {
        if bytes[index] != b'\\' || is_in_ranges(index, &protected) {
            index += 1;
            continue;
        }
        if bytes[index + 1] == b'\\' {
            index += 2;
            continue;
        }
        let (closing, dollars) = match bytes[index + 1] {
            b'(' => (b')', "$"),
            b'[' => (b']', "$$"),
            _ => {
                index += 1;
                continue;
            }
        };
        if let Some(close) = find_latex_close(bytes, index + 2, closing, &protected) {
            output.push_str(&markdown[copied..index]);
            output.push_str(dollars);
            output.push_str(&markdown[index + 2..close]);
            output.push_str(dollars);
            index = close + 2;
            copied = index;
        } else {
            index += 1;
        }
    }
    output.push_str(&markdown[copied..]);
    output
}

fn protected_markdown_ranges(markdown: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut block_start = None;
    let parser = Parser::new_ext(markdown, Options::empty());
    for (_, definition) in parser.reference_definitions().iter() {
        ranges.push(definition.span.clone());
    }
    for (event, range) in parser.into_offset_iter() {
        match event {
            Event::Start(Tag::CodeBlock(_)) => block_start = Some(range.start),
            Event::End(TagEnd::CodeBlock) => {
                if let Some(start) = block_start.take() {
                    ranges.push(start..range.end);
                }
            }
            Event::Start(Tag::Link { .. }) | Event::Start(Tag::Image { .. }) => {
                ranges.push(range);
            }
            Event::Code(_) => ranges.push(range),
            Event::Html(_) | Event::InlineHtml(_) => ranges.push(range),
            _ => {}
        }
    }
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && range.start <= previous.end
        {
            previous.end = previous.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn find_latex_close(
    bytes: &[u8],
    from: usize,
    closing: u8,
    protected: &[Range<usize>],
) -> Option<usize> {
    let mut index = from;
    while index + 1 < bytes.len() {
        // Math delimiters must not span a code construct, raw HTML, or a
        // Markdown link/image. In particular, `\\(` is a valid escaped `(`
        // inside a destination and must remain untouched.
        if is_in_ranges(index, protected) {
            return None;
        }
        if closing == b')' && matches!(bytes[index], b'\n' | b'\r') {
            return None;
        }
        if closing == b']' && begins_blank_line(bytes, index) {
            return None;
        }
        if bytes[index] == b'\\' {
            if bytes[index + 1] == b'\\' {
                index += 2;
                continue;
            }
            if bytes[index + 1] == closing {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

fn begins_blank_line(bytes: &[u8], index: usize) -> bool {
    if !matches!(bytes[index], b'\n' | b'\r') {
        return false;
    }
    let mut cursor = index + 1;
    if bytes[index] == b'\r' && bytes.get(cursor) == Some(&b'\n') {
        cursor += 1;
    }
    while matches!(bytes.get(cursor), Some(b' ' | b'\t')) {
        cursor += 1;
    }
    matches!(bytes.get(cursor), Some(b'\n' | b'\r'))
}

fn is_in_ranges(position: usize, ranges: &[Range<usize>]) -> bool {
    let insertion = ranges.partition_point(|range| range.start <= position);
    insertion > 0 && position < ranges[insertion - 1].end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().expect("temporary directory");
        let spec = directory.path().join("Decks").join("science.md");
        fs::create_dir_all(spec.parent().expect("spec parent")).expect("create deck directory");
        fs::write(&spec, "Q: Test\nA: Test").expect("write spec");
        (directory, spec)
    }

    #[test]
    fn renders_gfm_and_math() {
        let (directory, spec) = fixture();
        let markdown = concat!(
            "| $A$ | $\\neg B$ |\n|---|---|\n| 1 | 2 |\n\n",
            "~~gone~~\n\n- [x] done -- yes\n\n",
            "$x < y$ and \\(a+b\\)\n\n",
            "$$x^2$$\n\n\\[y^2\\]",
        );
        let html = markdown_to_html(markdown, directory.path(), &spec);
        assert!(html.contains("<table>"));
        assert!(html.contains("<span class=\"math math-inline\">\\neg B</span>"));
        assert!(html.contains("<del>gone</del>"));
        assert!(html.contains("type=\"checkbox\" checked=\"\""));
        assert!(html.contains('–'));
        assert!(html.contains("<span class=\"math math-inline\">x &lt; y</span>"));
        assert!(html.contains("<span class=\"math math-inline\">a+b</span>"));
        assert!(html.contains("<span class=\"math math-display\">x^2</span>"));
        assert!(html.contains("<span class=\"math math-display\">y^2</span>"));
    }

    #[test]
    fn code_does_not_become_math() {
        let (directory, spec) = fixture();
        let markdown = "`\\(inline\\)`\n\n```txt\n\\[block\\]\n```";
        let html = markdown_to_html(markdown, directory.path(), &spec);
        assert!(html.contains("<code>\\(inline\\)</code>"));
        assert!(html.contains("\\[block\\]"));
        assert!(!html.contains("math-inline"));
        assert!(!html.contains("math-display"));
    }

    #[test]
    fn latex_rewrite_does_not_change_links_or_media_paths() {
        let (directory, spec) = fixture();
        let media_path = spec.parent().expect("spec parent").join("(draft).png");
        fs::write(media_path, b"image").expect("write image");
        let markdown = concat!(
            "[inline](notes/\\(draft\\).md) ",
            "[reference][draft] ",
            "![image](\\(draft\\).png)\n\n",
            "[draft]: notes/\\(reference\\).md",
        );
        let html = markdown_to_html(markdown, directory.path(), &spec);
        assert!(html.contains("href=\"notes/(draft).md\""));
        assert!(html.contains("href=\"notes/(reference).md\""));
        assert!(html.contains("src=\"/file/Decks/%28draft%29.png\""));
        assert!(!html.contains("notes/$draft$.md"));
    }

    #[test]
    fn latex_delimiters_do_not_pair_across_paragraphs() {
        let (directory, spec) = fixture();
        let inline = markdown_to_html(
            "prefix \\(x\n\nnew paragraph\n\ny\\) suffix",
            directory.path(),
            &spec,
        );
        let display = markdown_to_html(
            "prefix \\[x\n\nnew paragraph\n\ny\\] suffix",
            directory.path(),
            &spec,
        );
        assert!(!inline.contains("math-inline"));
        assert!(!display.contains("math-display"));
    }

    #[test]
    fn raw_html_and_unsafe_links_are_neutralized() {
        let (directory, spec) = fixture();
        let markdown = concat!(
            "<script>alert('x')</script>\n\n",
            "[bad](javascript:alert(1)) ",
            "[good](https://example.com/?a=1&b=2)",
        );
        let html = markdown_to_html(markdown, directory.path(), &spec);
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(!html.contains("href=\"javascript:"));
        assert!(html.contains("bad"));
        assert!(html.contains(
            "href=\"https://example.com/?a=1&amp;b=2\" target=\"_blank\" rel=\"noopener noreferrer\""
        ));
    }

    #[test]
    fn local_links_remain_same_context() {
        let (directory, spec) = fixture();
        let html = markdown_to_html(
            "[section](#answer) [note](notes/topic.md)",
            directory.path(),
            &spec,
        );
        assert!(html.contains("<a href=\"#answer\">section</a>"));
        assert!(html.contains("<a href=\"notes/topic.md\">note</a>"));
        assert!(!html.contains("target=\"_blank\""));
    }

    #[test]
    fn resolves_collection_and_spec_relative_media() {
        let (directory, spec) = fixture();
        fs::create_dir_all(directory.path().join("Images")).expect("create images");
        fs::write(directory.path().join("Images/root image.png"), b"png").expect("write image");
        fs::write(
            spec.parent().expect("spec parent").join("sound.mp3"),
            b"mp3",
        )
        .expect("write audio");

        let html = markdown_to_html(
            "![plot](<@/Images/root image.png>)\n\n![listen](sound.mp3)",
            directory.path(),
            &spec,
        );
        assert!(html.contains("src=\"/file/Images/root%20image.png\""));
        assert!(html.contains("alt=\"plot\""));
        assert!(html.contains("<audio controls"));
        assert!(html.contains("src=\"/file/Decks/sound.mp3\""));
    }

    #[test]
    fn supports_safe_parent_resolution_within_collection() {
        let (directory, spec) = fixture();
        fs::write(directory.path().join("shared.webp"), b"image").expect("write image");
        let media = resolve_media_reference(directory.path(), &spec, "../shared.webp")
            .expect("resolve parent path");
        assert_eq!(media.url, "/file/shared.webp");
        assert_eq!(media.kind, MediaKind::Image);
    }

    #[test]
    fn bad_media_is_a_visible_escaped_fallback() {
        let (directory, spec) = fixture();
        let html = markdown_to_html("![<unsafe>](../../outside.png)", directory.path(), &spec);
        assert!(html.contains("class=\"media-fallback\""));
        assert!(html.contains("&lt;unsafe&gt;"));
        assert!(!html.contains("<unsafe>"));
    }

    #[test]
    fn media_request_is_validated_and_typed() {
        let (directory, _) = fixture();
        let path = directory.path().join("clip.webm");
        fs::write(&path, b"video").expect("write video");
        let file = validate_media_request(directory.path(), "clip.webm").expect("valid media");
        assert_eq!(file.path, path.canonicalize().expect("canonical path"));
        assert_eq!(file.content_type, "video/webm");
        assert_eq!(file.kind, MediaKind::Video);
        assert_eq!(
            validate_media_request(directory.path(), "../clip.webm"),
            Err(MediaError::ParentTraversal)
        );
        assert_eq!(
            validate_media_request(directory.path(), "/etc/passwd"),
            Err(MediaError::Absolute)
        );
    }

    #[cfg(unix)]
    #[test]
    fn media_request_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let (directory, _) = fixture();
        let target = directory.path().join("target.png");
        let link = directory.path().join("link.png");
        fs::write(&target, b"image").expect("write target");
        symlink(&target, &link).expect("create symlink");
        assert_eq!(
            validate_media_request(directory.path(), "link.png"),
            Err(MediaError::Symlink)
        );
    }

    #[test]
    fn media_request_rejects_directories_and_unknown_types() {
        let (directory, _) = fixture();
        fs::write(directory.path().join("notes.txt"), b"text").expect("write text");
        assert_eq!(
            validate_media_request(directory.path(), "Decks"),
            Err(MediaError::NotFile)
        );
        assert_eq!(
            validate_media_request(directory.path(), "notes.txt"),
            Err(MediaError::UnsupportedType)
        );
    }

    #[test]
    fn percent_decodes_incoming_file_paths() {
        let (directory, _) = fixture();
        fs::write(directory.path().join("wide shot.jpg"), b"image").expect("write image");
        let file = validate_media_request(directory.path(), "wide%20shot.jpg")
            .expect("decode request path");
        assert_eq!(file.content_type, "image/jpeg");
    }

    #[test]
    fn encoded_traversal_and_absolute_requests_are_rejected() {
        let (directory, _) = fixture();
        assert_eq!(
            validate_media_request(directory.path(), "%2e%2e/secret.png"),
            Err(MediaError::ParentTraversal)
        );
        assert_eq!(
            validate_media_request(directory.path(), "%2E%2E%2Fsecret.png"),
            Err(MediaError::ParentTraversal)
        );
        assert_eq!(
            validate_media_request(directory.path(), ".%2e/secret.png"),
            Err(MediaError::ParentTraversal)
        );
        assert_eq!(
            validate_media_request(directory.path(), "%2Fetc/passwd"),
            Err(MediaError::Absolute)
        );
        assert_eq!(
            validate_media_request(directory.path(), "%252e%252e/secret.png"),
            Err(MediaError::NotFound)
        );
    }

    #[cfg(unix)]
    #[test]
    fn authored_symlink_cannot_escape_collection() {
        use std::os::unix::fs::symlink;

        let (directory, spec) = fixture();
        let outside = tempfile::tempdir().expect("outside temporary directory");
        let target = outside.path().join("outside.png");
        fs::write(&target, b"image").expect("write outside image");
        let link = spec.parent().expect("spec parent").join("outside.png");
        symlink(&target, link).expect("create outside symlink");
        assert_eq!(
            resolve_media_reference(directory.path(), &spec, "outside.png"),
            Err(MediaError::OutsideCollection)
        );
    }

    #[test]
    fn inline_rendering_removes_only_a_paragraph() {
        let (directory, spec) = fixture();
        assert_eq!(
            markdown_to_html_inline("some **bold** text", directory.path(), &spec),
            "some <strong>bold</strong> text"
        );
        assert_eq!(
            markdown_to_html_inline("# Heading", directory.path(), &spec),
            "<h1>Heading</h1>\n"
        );
    }
}
