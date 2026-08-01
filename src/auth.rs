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

//! Launch-scoped access control for the local review server.
//!
//! The printed bootstrap URL is a bearer capability. It is exchanged for an
//! HttpOnly cookie and is valid only for the lifetime of one server process.

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

use axum::http::HeaderMap;
use axum::http::HeaderValue;
use axum::http::header::COOKIE;
use axum::http::header::HOST;
use axum::http::header::ORIGIN;
use axum::http::uri::Authority;
use url::Url;

use crate::error::ErrorReport;
use crate::error::Fallible;
use crate::error::fail;

pub const BOOTSTRAP_PATH: &str = "/_hashdrills/auth";

const TOKEN_BYTES: usize = 32;
const COOKIE_PREFIX: &str = "hashdrills_";

/// Request-time access policy shared by every web route.
#[derive(Clone)]
pub struct AccessControl {
    enabled: bool,
    bypass_security: bool,
    token_digest: [u8; 32],
    csrf_token: Arc<str>,
    csrf_digest: [u8; 32],
    cookie_name: Arc<str>,
    base_path: Arc<str>,
    origins: Arc<Vec<AllowedOrigin>>,
}

/// URLs and request policy generated for one server launch.
pub struct LaunchAccess {
    pub control: AccessControl,
    pub browser_url: String,
    pub public_url: Option<String>,
    pub readiness_host: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AllowedOrigin {
    serialized: String,
    host: String,
    port: u16,
    secure: bool,
}

impl LaunchAccess {
    /// Generate a fresh access capability and validate the advertised origins.
    pub fn new(
        enabled: bool,
        bind_host: &str,
        bound_address: SocketAddr,
        public_url: Option<&str>,
    ) -> Fallible<Self> {
        if !enabled && !bound_address.ip().is_loopback() {
            return fail("--no-auth may only be used with a loopback bind address");
        }

        let wildcard = is_wildcard(bind_host);
        if enabled && wildcard && public_url.is_none() {
            return fail("--public-url is required when --host is a wildcard address");
        }

        let readiness_host = if wildcard {
            "127.0.0.1".to_string()
        } else {
            unbracket(bind_host).to_string()
        };
        let local_origin = parse_origin(&format!(
            "http://{}:{}",
            display_host(&readiness_host),
            bound_address.port()
        ))?;
        let advertised_origin = match public_url {
            Some(value) => parse_origin(value)?,
            None => local_origin.clone(),
        };
        if local_origin.host == advertised_origin.host
            && local_origin.port == advertised_origin.port
            && local_origin.secure != advertised_origin.secure
        {
            return fail(
                "--public-url cannot use the local bind authority with a different URL scheme",
            );
        }

        let mut origins = vec![local_origin.clone()];
        if !origins.contains(&advertised_origin) {
            origins.push(advertised_origin.clone());
        }

        let token = random_hex(TOKEN_BYTES)?;
        let csrf_token = random_hex(TOKEN_BYTES)?;
        let base_path = format!("/_hashdrills/session/{}", random_hex(16)?);
        let token_digest = digest(&token);
        let csrf_digest = digest(&csrf_token);
        let cookie_name = format!("{COOKIE_PREFIX}{}", &hex(&token_digest)[..12]);
        let control = AccessControl {
            enabled,
            bypass_security: false,
            token_digest,
            csrf_token: Arc::from(csrf_token),
            csrf_digest,
            cookie_name: Arc::from(cookie_name),
            base_path: Arc::from(base_path),
            origins: Arc::new(origins),
        };

        let browser_url = if enabled {
            bootstrap_url(&local_origin.serialized, &token)?
        } else {
            format!("{}{}/", local_origin.serialized, control.base_path)
        };
        let advertised_url = if enabled {
            bootstrap_url(&advertised_origin.serialized, &token)?
        } else {
            format!("{}{}/", advertised_origin.serialized, control.base_path)
        };
        let public_url = (advertised_url != browser_url).then_some(advertised_url);

        Ok(Self {
            control,
            browser_url,
            public_url,
            readiness_host,
        })
    }
}

impl AccessControl {
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn bypasses_security(&self) -> bool {
        self.bypass_security
    }

    pub fn base_path(&self) -> &str {
        &self.base_path
    }

    pub fn root_path(&self) -> String {
        format!("{}/", self.base_path)
    }

    /// Validate the request authority against the configured local/public URLs.
    pub fn host_is_allowed(&self, headers: &HeaderMap) -> bool {
        self.matching_host(headers).is_some()
    }

    /// Require the browser's POST Origin to describe the same allowed origin
    /// as the request Host. Missing, duplicate, and `null` origins fail closed.
    pub fn post_origin_is_allowed(&self, headers: &HeaderMap) -> bool {
        if self.bypass_security {
            return true;
        }
        let Some(authority) = single_header(headers, HOST)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| Authority::from_str(value).ok())
        else {
            return false;
        };
        let Some(origin) = single_header(headers, ORIGIN)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| parse_origin(value).ok())
        else {
            return false;
        };
        self.origins.iter().any(|allowed| {
            allowed.matches_authority(&authority) && allowed.serialized == origin.serialized
        })
    }

    pub fn token_is_valid(&self, token: &str) -> bool {
        self.enabled && constant_time_eq(&self.token_digest, &digest(token))
    }

    pub fn cookie_is_valid(&self, headers: &HeaderMap) -> bool {
        if !self.enabled {
            return true;
        }
        let mut found = None;
        for header in headers.get_all(COOKIE) {
            let Ok(header) = header.to_str() else {
                return false;
            };
            for pair in header.split(';') {
                let Some((name, value)) = pair.trim().split_once('=') else {
                    continue;
                };
                if name == self.cookie_name.as_ref() {
                    if found.replace(value).is_some() {
                        return false;
                    }
                }
            }
        }
        found.is_some_and(|value| self.token_is_valid(value))
    }

    /// Construct the host-only session cookie after a successful bootstrap.
    pub fn bootstrap_cookie(&self, token: &str, headers: &HeaderMap) -> Option<HeaderValue> {
        if !self.token_is_valid(token) {
            return None;
        }
        let secure = self
            .matching_host(headers)?
            .iter()
            .any(|origin| origin.secure);
        let secure_attribute = if secure { "; Secure" } else { "" };
        HeaderValue::from_str(&format!(
            "{}={token}; Path={}/; HttpOnly; SameSite=Strict{secure_attribute}",
            self.cookie_name, self.base_path
        ))
        .ok()
    }

    pub fn csrf_token(&self) -> &str {
        &self.csrf_token
    }

    pub fn csrf_is_valid(&self, token: &str) -> bool {
        self.bypass_security || constant_time_eq(&self.csrf_digest, &digest(token))
    }

    fn matching_host<'a>(&'a self, headers: &HeaderMap) -> Option<Vec<&'a AllowedOrigin>> {
        let authority = single_header(headers, HOST)?
            .to_str()
            .ok()
            .and_then(|value| Authority::from_str(value).ok())?;
        let matching: Vec<_> = self
            .origins
            .iter()
            .filter(|origin| origin.matches_authority(&authority))
            .collect();
        (!matching.is_empty()).then_some(matching)
    }

    #[cfg(test)]
    pub(crate) fn disabled_for_tests() -> Self {
        Self {
            enabled: false,
            bypass_security: true,
            token_digest: [0; 32],
            csrf_token: Arc::from(""),
            csrf_digest: [0; 32],
            cookie_name: Arc::from("hashdrills_test"),
            base_path: Arc::from(""),
            origins: Arc::new(Vec::new()),
        }
    }
}

impl AllowedOrigin {
    fn matches_authority(&self, authority: &Authority) -> bool {
        let host = normalize_host(authority.host());
        if host != self.host {
            return false;
        }
        match authority.port_u16() {
            Some(port) => port == self.port,
            None => self.port == if self.secure { 443 } else { 80 },
        }
    }
}

fn single_header(
    headers: &HeaderMap,
    name: axum::http::header::HeaderName,
) -> Option<&HeaderValue> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn parse_origin(value: &str) -> Fallible<AllowedOrigin> {
    let url = Url::parse(value)
        .map_err(|error| ErrorReport::new(format!("invalid web origin '{value}': {error}")))?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return fail(format!(
            "web origin must be an http(s) origin without credentials, path, query, or fragment: {value}"
        ));
    }
    let Some(host) = url.host_str() else {
        return fail(format!("web origin has no host: {value}"));
    };
    let host = normalize_host(host);
    if matches!(host.as_str(), "0.0.0.0" | "::") {
        return fail("a wildcard bind address cannot be used as a public web origin");
    }
    let Some(port) = url.port_or_known_default() else {
        return fail(format!("web origin has no usable port: {value}"));
    };
    Ok(AllowedOrigin {
        serialized: url.origin().ascii_serialization(),
        host,
        port,
        secure: url.scheme() == "https",
    })
}

fn bootstrap_url(origin: &str, token: &str) -> Fallible<String> {
    let mut url = Url::parse(origin)
        .map_err(|error| ErrorReport::new(format!("invalid web origin '{origin}': {error}")))?;
    url.set_path(BOOTSTRAP_PATH);
    url.query_pairs_mut().append_pair("access_token", token);
    Ok(url.into())
}

fn is_wildcard(host: &str) -> bool {
    matches!(host, "0.0.0.0" | "::" | "[::]")
}

fn unbracket(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host)
}

fn display_host(host: &str) -> String {
    if host.contains(':') {
        format!("[{}]", unbracket(host))
    } else {
        host.to_string()
    }
}

fn normalize_host(host: &str) -> String {
    unbracket(host).trim_end_matches('.').to_ascii_lowercase()
}

fn random_hex(bytes: usize) -> Fallible<String> {
    let mut random = vec![0_u8; bytes];
    getrandom::fill(&mut random).map_err(|error| {
        ErrorReport::new(format!("could not generate web access token: {error}"))
    })?;
    Ok(hex(&random))
}

fn digest(value: &str) -> [u8; 32] {
    *blake3::hash(value.as_bytes()).as_bytes()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    let mut difference = 0_u8;
    for (left, right) in left.iter().zip(right) {
        difference |= left ^ right;
    }
    difference == 0
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;
    use axum::http::HeaderValue;
    use axum::http::header::COOKIE;
    use axum::http::header::HOST;
    use axum::http::header::ORIGIN;

    use super::*;

    #[test]
    fn launch_generates_a_256_bit_capability_and_strict_cookie() {
        let launch =
            LaunchAccess::new(true, "100.64.0.7", "100.64.0.7:8000".parse().unwrap(), None)
                .unwrap();
        let url = Url::parse(&launch.browser_url).unwrap();
        let token = url
            .query_pairs()
            .find_map(|(name, value)| (name == "access_token").then(|| value.into_owned()))
            .unwrap();
        assert_eq!(token.len(), 64);
        assert!(token.bytes().all(|byte| byte.is_ascii_hexdigit()));

        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("100.64.0.7:8000"));
        let cookie = launch
            .control
            .bootstrap_cookie(&token, &headers)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let cookie_path = format!("; Path={}/", launch.control.base_path());
        assert!(cookie.contains(&cookie_path));
        assert!(
            launch
                .control
                .base_path()
                .starts_with("/_hashdrills/session/")
        );
        assert_eq!(
            launch
                .control
                .base_path()
                .trim_start_matches("/_hashdrills/session/")
                .len(),
            32
        );
        assert!(cookie.contains("; HttpOnly"));
        assert!(cookie.contains("; SameSite=Strict"));
        assert!(!cookie.contains("; Domain="));
        assert!(!cookie.contains("; Secure"));

        let cookie_pair = cookie.split(';').next().unwrap();
        headers.insert(COOKIE, cookie_pair.parse().unwrap());
        assert!(launch.control.cookie_is_valid(&headers));
    }

    #[test]
    fn wildcard_requires_public_url_and_no_auth_requires_loopback() {
        assert!(LaunchAccess::new(true, "0.0.0.0", "0.0.0.0:8000".parse().unwrap(), None).is_err());
        assert!(
            LaunchAccess::new(
                false,
                "0.0.0.0",
                "0.0.0.0:8000".parse().unwrap(),
                Some("http://100.64.0.7:8000")
            )
            .is_err()
        );
        assert!(
            LaunchAccess::new(false, "127.0.0.1", "127.0.0.1:8000".parse().unwrap(), None).is_ok()
        );
    }

    #[test]
    fn no_auth_keeps_host_origin_and_csrf_protection() {
        let launch =
            LaunchAccess::new(false, "127.0.0.1", "127.0.0.1:8000".parse().unwrap(), None).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("127.0.0.1:8000"));
        assert!(launch.control.host_is_allowed(&headers));
        assert!(!launch.control.post_origin_is_allowed(&headers));
        headers.insert(ORIGIN, HeaderValue::from_static("http://127.0.0.1:8000"));
        assert!(launch.control.post_origin_is_allowed(&headers));
        assert!(launch.control.csrf_is_valid(launch.control.csrf_token()));
        assert!(!launch.control.csrf_is_valid("wrong"));

        headers.insert(HOST, HeaderValue::from_static("attacker.test"));
        assert!(!launch.control.host_is_allowed(&headers));
    }

    #[test]
    fn host_origin_cookie_and_csrf_fail_closed() {
        let launch = LaunchAccess::new(
            true,
            "0.0.0.0",
            "0.0.0.0:8000".parse().unwrap(),
            Some("https://study.example"),
        )
        .unwrap();
        let token = Url::parse(launch.public_url.as_ref().unwrap())
            .unwrap()
            .query_pairs()
            .find_map(|(name, value)| (name == "access_token").then(|| value.into_owned()))
            .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("study.example"));
        headers.insert(ORIGIN, HeaderValue::from_static("https://study.example"));
        assert!(launch.control.host_is_allowed(&headers));
        assert!(launch.control.post_origin_is_allowed(&headers));
        assert!(launch.control.token_is_valid(&token));
        assert!(!launch.control.token_is_valid("wrong"));
        assert!(
            launch
                .control
                .bootstrap_cookie(&token, &headers)
                .unwrap()
                .to_str()
                .unwrap()
                .contains("; Secure")
        );
        assert!(launch.control.csrf_is_valid(launch.control.csrf_token()));
        assert!(!launch.control.csrf_is_valid("wrong"));

        headers.insert(ORIGIN, HeaderValue::from_static("https://evil.example"));
        assert!(!launch.control.post_origin_is_allowed(&headers));
        headers.insert(HOST, HeaderValue::from_static("evil.example"));
        assert!(!launch.control.host_is_allowed(&headers));
    }

    #[test]
    fn public_url_rejects_non_origins() {
        for value in [
            "ftp://example.test",
            "http://user@example.test",
            "http://example.test/path",
            "http://example.test/?query",
            "http://example.test/#fragment",
            "http://0.0.0.0:8000",
        ] {
            assert!(
                LaunchAccess::new(
                    true,
                    "127.0.0.1",
                    "127.0.0.1:8000".parse().unwrap(),
                    Some(value),
                )
                .is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn public_url_rejects_an_ambiguous_http_https_authority() {
        assert!(
            LaunchAccess::new(
                true,
                "127.0.0.1",
                "127.0.0.1:8000".parse().unwrap(),
                Some("https://127.0.0.1:8000"),
            )
            .is_err()
        );
    }
}
