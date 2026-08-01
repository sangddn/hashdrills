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

//! Stable identifiers for user-requested session-transcript downloads.

/// Version of the Markdown session-transcript format.
pub const SESSION_LOG_FORMAT_VERSION: u32 = 1;

/// Transcript kind stored beside [`SESSION_LOG_FORMAT_VERSION`].
pub const SESSION_LOG_KIND: &str = "session_transcript";

/// Frontmatter key identifying a Hashdrills session transcript.
pub const SESSION_LOG_KIND_MARKER_KEY: &str = "hashdrills_session_log_kind";

/// Frontmatter key selecting the transcript representation version.
pub const SESSION_LOG_VERSION_MARKER_KEY: &str = "hashdrills_session_log_format_version";
