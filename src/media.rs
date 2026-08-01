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

//! Bounded-memory, race-resistant serving for validated collection media.

use std::fs::File;
use std::io;
use std::path::Path;

use axum::body::Body;
use axum::http::HeaderMap;
use axum::http::HeaderValue;
use axum::http::StatusCode;
use axum::http::header::ACCEPT_RANGES;
use axum::http::header::CACHE_CONTROL;
use axum::http::header::CONTENT_LENGTH;
use axum::http::header::CONTENT_RANGE;
use axum::http::header::CONTENT_SECURITY_POLICY;
use axum::http::header::CONTENT_TYPE;
use axum::http::header::RANGE;
use axum::http::header::X_CONTENT_TYPE_OPTIONS;
use axum::response::Response;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncSeekExt;
use tokio::io::SeekFrom;
use tokio_util::io::ReaderStream;

use crate::render::MediaFile;
use crate::render::MediaIdentity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestedRange {
    Closed { start: u64, end: u64 },
    From(u64),
    Suffix(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedRange {
    start: u64,
    end: u64,
}

impl ResolvedRange {
    fn length(self) -> u64 {
        self.end - self.start + 1
    }
}

/// Stream a previously validated media file with single-range support.
///
/// `headers` is inspected before the file is opened. Multiple Range header
/// fields and multipart byte ranges are deliberately rejected: supporting one
/// range is sufficient for browser audio/video seeking and keeps this local
/// endpoint small and deterministic.
pub async fn serve(file: MediaFile, headers: &HeaderMap) -> Response {
    let requested_range = parse_range_header(headers);

    let path = file.path.clone();
    let expected_identity = file.identity.clone();
    let opened =
        tokio::task::spawn_blocking(move || open_verified(&path, &expected_identity)).await;
    let (opened, length) = match opened {
        Ok(Ok(opened)) => opened,
        Ok(Err(_)) | Err(_) => return not_found(),
    };

    let requested_range = match requested_range {
        Ok(range) => range,
        Err(()) => return range_not_satisfiable(length),
    };

    let resolved_range = match requested_range {
        Some(requested) => match resolve_range(requested, length) {
            Ok(range) => Some(range),
            Err(()) => return range_not_satisfiable(length),
        },
        None => None,
    };

    let (status, start, response_length) = match resolved_range {
        Some(range) => (StatusCode::PARTIAL_CONTENT, range.start, range.length()),
        None => (StatusCode::OK, 0, length),
    };

    let mut opened = tokio::fs::File::from_std(opened);
    if start != 0 && opened.seek(SeekFrom::Start(start)).await.is_err() {
        return not_found();
    }
    let stream = ReaderStream::new(opened.take(response_length));
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    let response_headers = response.headers_mut();
    response_headers.insert(CONTENT_TYPE, HeaderValue::from_static(file.content_type));
    response_headers.insert(CACHE_CONTROL, HeaderValue::from_static("private, no-store"));
    response_headers.insert(ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    response_headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    response_headers.insert(
        CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("sandbox; default-src 'none'; style-src 'unsafe-inline'"),
    );
    insert_u64_header(response_headers, CONTENT_LENGTH, response_length);
    if let Some(range) = resolved_range {
        insert_string_header(
            response_headers,
            CONTENT_RANGE,
            format!("bytes {}-{}/{}", range.start, range.end, length),
        );
    }
    response
}

fn parse_range_header(headers: &HeaderMap) -> Result<Option<RequestedRange>, ()> {
    let mut values = headers.get_all(RANGE).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(());
    }
    let value = value.to_str().map_err(|_| ())?.trim();
    let (unit, value) = value.split_once('=').ok_or(())?;
    if !unit.trim().eq_ignore_ascii_case("bytes") {
        return Err(());
    }
    let value = value.trim();
    if value.is_empty() || value.contains(',') {
        return Err(());
    }
    let (start, end) = value.split_once('-').ok_or(())?;
    match (start.trim(), end.trim()) {
        ("", "") => Err(()),
        ("", suffix) => {
            let suffix = parse_u64(suffix)?;
            (suffix != 0)
                .then_some(Some(RequestedRange::Suffix(suffix)))
                .ok_or(())
        }
        (start, "") => Ok(Some(RequestedRange::From(parse_u64(start)?))),
        (start, end) => {
            let start = parse_u64(start)?;
            let end = parse_u64(end)?;
            (start <= end)
                .then_some(Some(RequestedRange::Closed { start, end }))
                .ok_or(())
        }
    }
}

fn parse_u64(value: &str) -> Result<u64, ()> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(());
    }
    value.parse().map_err(|_| ())
}

fn resolve_range(requested: RequestedRange, length: u64) -> Result<ResolvedRange, ()> {
    if length == 0 {
        return Err(());
    }
    match requested {
        RequestedRange::Closed { start, end } if start < length => Ok(ResolvedRange {
            start,
            end: end.min(length - 1),
        }),
        RequestedRange::From(start) if start < length => Ok(ResolvedRange {
            start,
            end: length - 1,
        }),
        RequestedRange::Suffix(suffix) => Ok(ResolvedRange {
            start: length.saturating_sub(suffix),
            end: length - 1,
        }),
        RequestedRange::Closed { .. } | RequestedRange::From(_) => Err(()),
    }
}

fn open_verified(path: &Path, expected: &MediaIdentity) -> io::Result<(File, u64)> {
    let before = std::fs::symlink_metadata(path)?;
    ensure_expected_regular(&before, expected)?;
    ensure_still_canonical(path)?;

    let opened = open_readonly(path)?;
    let opened_metadata = opened.metadata()?;
    ensure_expected_regular(&opened_metadata, expected)?;

    // Both checks happen after opening. The identity comparison detects a
    // final-file or parent-directory swap; re-canonicalization also rejects an
    // intermediate component replaced by a symlink.
    let after = std::fs::symlink_metadata(path)?;
    ensure_expected_regular(&after, expected)?;
    ensure_still_canonical(path)?;
    if !same_file_identity(&opened_metadata, &after) {
        return Err(io::Error::other("media path changed while opening"));
    }
    let length = opened_metadata.len();
    Ok((opened, length))
}

#[cfg(unix)]
fn open_readonly(path: &Path) -> io::Result<File> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        // O_NOFOLLOW rejects a final-component symlink at open time.
        // O_NONBLOCK keeps a last-moment FIFO/device swap from tying up a
        // blocking worker before the regular-file identity checks can run.
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
}

#[cfg(not(unix))]
fn open_readonly(path: &Path) -> io::Result<File> {
    File::open(path)
}

fn ensure_expected_regular(
    metadata: &std::fs::Metadata,
    expected: &MediaIdentity,
) -> io::Result<()> {
    if metadata.file_type().is_symlink() || !metadata.is_file() || !expected.matches(metadata) {
        return Err(io::Error::other("media identity changed"));
    }
    Ok(())
}

fn ensure_still_canonical(path: &Path) -> io::Result<()> {
    if path.canonicalize()? != path {
        return Err(io::Error::other("media path is no longer canonical"));
    }
    Ok(())
}

#[cfg(unix)]
fn same_file_identity(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
        && left.created().ok() == right.created().ok()
}

fn range_not_satisfiable(length: u64) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::RANGE_NOT_SATISFIABLE;
    let headers = response.headers_mut();
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("private, no-store"));
    headers.insert(ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    insert_u64_header(headers, CONTENT_LENGTH, 0);
    insert_string_header(headers, CONTENT_RANGE, format!("bytes */{length}"));
    response
}

fn not_found() -> Response {
    let mut response = Response::new(Body::from("Not Found"));
    *response.status_mut() = StatusCode::NOT_FOUND;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("private, no-store"));
    response
}

fn insert_u64_header(headers: &mut HeaderMap, name: axum::http::header::HeaderName, value: u64) {
    insert_string_header(headers, name, value.to_string());
}

fn insert_string_header(
    headers: &mut HeaderMap,
    name: axum::http::header::HeaderName,
    value: String,
) {
    if let Ok(value) = HeaderValue::from_str(&value) {
        headers.insert(name, value);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use axum::body::Body;
    use axum::http::HeaderMap;
    use axum::http::HeaderValue;
    use axum::http::StatusCode;
    use axum::http::header::ACCEPT_RANGES;
    use axum::http::header::CACHE_CONTROL;
    use axum::http::header::CONTENT_LENGTH;
    use axum::http::header::CONTENT_RANGE;
    use axum::http::header::RANGE;
    use http_body_util::BodyExt;
    use tempfile::TempDir;

    use super::RequestedRange;
    use super::parse_range_header;
    use super::serve;
    use crate::render::MediaFile;
    use crate::render::validate_media_request;

    fn fixture(contents: &[u8]) -> (TempDir, MediaFile) {
        let directory = tempfile::tempdir().expect("tempdir");
        fs::write(directory.path().join("media.mp4"), contents).expect("write media");
        let media = validate_media_request(directory.path(), "media.mp4").expect("validate");
        (directory, media)
    }

    fn headers_with_range(value: &'static str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(RANGE, HeaderValue::from_static(value));
        headers
    }

    async fn body_bytes(body: Body) -> Vec<u8> {
        axum::body::to_bytes(body, usize::MAX)
            .await
            .expect("body bytes")
            .to_vec()
    }

    #[test]
    fn parses_closed_open_and_suffix_ranges() {
        assert_eq!(
            parse_range_header(&headers_with_range("bytes=2-5")),
            Ok(Some(RequestedRange::Closed { start: 2, end: 5 }))
        );
        assert_eq!(
            parse_range_header(&headers_with_range("bytes=2-")),
            Ok(Some(RequestedRange::From(2)))
        );
        assert_eq!(
            parse_range_header(&headers_with_range("bytes=-4")),
            Ok(Some(RequestedRange::Suffix(4)))
        );
    }

    #[tokio::test]
    async fn serves_full_content_with_private_streaming_headers() {
        let (_directory, media) = fixture(b"0123456789");
        let response = serve(media, &HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_LENGTH], "10");
        assert_eq!(response.headers()[ACCEPT_RANGES], "bytes");
        assert_eq!(response.headers()[CACHE_CONTROL], "private, no-store");
        assert_eq!(body_bytes(response.into_body()).await, b"0123456789");
    }

    #[tokio::test]
    async fn serves_each_supported_single_range_shape() {
        let cases = [
            ("bytes=2-5", "bytes 2-5/10", b"2345".as_slice()),
            ("bytes=7-", "bytes 7-9/10", b"789".as_slice()),
            ("bytes=-4", "bytes 6-9/10", b"6789".as_slice()),
            ("bytes=8-99", "bytes 8-9/10", b"89".as_slice()),
        ];
        for (request_range, content_range, expected) in cases {
            let (_directory, media) = fixture(b"0123456789");
            let response = serve(media, &headers_with_range(request_range)).await;
            assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
            assert_eq!(response.headers()[CONTENT_RANGE], content_range);
            assert_eq!(
                response.headers()[CONTENT_LENGTH],
                expected.len().to_string()
            );
            assert_eq!(body_bytes(response.into_body()).await, expected);
        }
    }

    #[tokio::test]
    async fn rejects_invalid_multiple_and_unsatisfiable_ranges() {
        let requests = [
            "items=0-1",
            "bytes=",
            "bytes=5-2",
            "bytes=0-1,4-5",
            "bytes=99-",
        ];
        for request_range in requests {
            let (_directory, media) = fixture(b"0123456789");
            let response = serve(media, &headers_with_range(request_range)).await;
            assert_eq!(response.status(), StatusCode::RANGE_NOT_SATISFIABLE);
            assert_eq!(response.headers()[CONTENT_RANGE], "bytes */10");
            assert_eq!(response.headers()[CONTENT_LENGTH], "0");
        }

        let (_directory, media) = fixture(b"0123456789");
        let mut headers = HeaderMap::new();
        headers.append(RANGE, HeaderValue::from_static("bytes=0-1"));
        headers.append(RANGE, HeaderValue::from_static("bytes=4-5"));
        let response = serve(media, &headers).await;
        assert_eq!(response.status(), StatusCode::RANGE_NOT_SATISFIABLE);
        assert_eq!(response.headers()[CONTENT_RANGE], "bytes */10");
    }

    #[tokio::test]
    async fn large_file_body_arrives_in_bounded_chunks() {
        let contents = vec![0x5a; 4 * 1024 * 1024];
        let (_directory, media) = fixture(&contents);
        let response = serve(media, &HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::OK);

        let mut body = response.into_body();
        let first = body
            .frame()
            .await
            .expect("first frame")
            .expect("stream frame");
        let first_length = first.data_ref().expect("data frame").len();
        assert!(first_length < contents.len());
        let mut received = first_length;
        while let Some(frame) = body.frame().await {
            let frame = frame.expect("stream frame");
            if let Some(data) = frame.data_ref() {
                received += data.len();
            }
        }
        assert_eq!(received, contents.len());
    }

    #[tokio::test]
    async fn replacement_after_validation_is_not_served() {
        let (directory, media) = fixture(b"expected");
        let original = directory.path().join("media.mp4");
        fs::rename(&original, directory.path().join("original.mp4")).expect("preserve inode");
        fs::write(&original, b"attacker-controlled replacement").expect("replace file");

        let response = serve(media, &HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(body_bytes(response.into_body()).await, b"Not Found");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn final_symlink_swap_after_validation_is_not_served() {
        use std::os::unix::fs::symlink;

        let (directory, media) = fixture(b"expected");
        let original = directory.path().join("media.mp4");
        fs::rename(&original, directory.path().join("original.mp4")).expect("preserve inode");
        let outside = directory.path().join("outside.mp4");
        fs::write(&outside, b"outside").expect("write outside");
        symlink(&outside, &original).expect("replace with symlink");

        let response = serve(media, &HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn intermediate_symlink_swap_after_validation_is_not_served() {
        use std::os::unix::fs::symlink;

        let collection = tempfile::tempdir().expect("collection");
        let outside = tempfile::tempdir().expect("outside");
        let deck = collection.path().join("deck");
        fs::create_dir(&deck).expect("create deck");
        fs::write(deck.join("media.mp4"), b"expected").expect("write media");
        let media = validate_media_request(collection.path(), "deck/media.mp4").expect("validate");

        fs::rename(&deck, collection.path().join("original-deck")).expect("preserve deck");
        fs::write(outside.path().join("media.mp4"), b"outside").expect("write outside");
        symlink(outside.path(), &deck).expect("replace parent with symlink");

        let response = serve(media, &HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
