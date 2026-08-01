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

//! Embedded third-party assets used by the drill UI.
//!
//! Keeping these assets in the binary makes Markdown rendering work without a
//! CDN and gives the browser stable, cacheable URLs.

use axum::extract::Path;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderName, StatusCode};

pub const KATEX_CSS_URL: &str = "/assets/katex/katex.css";
pub const KATEX_JS_URL: &str = "/assets/katex/katex.js";
pub const KATEX_MHCHEM_JS_URL: &str = "/assets/katex/mhchem.js";
pub const KATEX_FONT_ROUTE: &str = "/assets/katex/fonts/{path}";

pub const HIGHLIGHT_CSS_URL: &str = "/assets/highlight/highlight.css";
pub const HIGHLIGHT_JS_URL: &str = "/assets/highlight/highlight.js";

const CACHE_CONTROL_IMMUTABLE: &str = "public, max-age=604800, immutable";
const CACHE_CONTROL_NOT_FOUND: &str = "no-cache";

pub type StaticAssetResponse = (StatusCode, [(HeaderName, &'static str); 2], &'static [u8]);

pub async fn katex_css_handler() -> StaticAssetResponse {
    asset_response(
        "text/css; charset=utf-8",
        include_bytes!("../vendor/katex/katex.min.css"),
    )
}

pub async fn katex_js_handler() -> StaticAssetResponse {
    asset_response(
        "text/javascript; charset=utf-8",
        include_bytes!("../vendor/katex/katex.min.js"),
    )
}

pub async fn katex_mhchem_js_handler() -> StaticAssetResponse {
    asset_response(
        "text/javascript; charset=utf-8",
        include_bytes!("../vendor/katex/contrib/mhchem.min.js"),
    )
}

pub async fn katex_font_handler(Path(path): Path<String>) -> StaticAssetResponse {
    // Do not turn this into a filesystem lookup. The explicit match is both an
    // allowlist and protection against path traversal.
    let bytes: &'static [u8] = match path.as_str() {
        "KaTeX_AMS-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_AMS-Regular.woff2")
        }
        "KaTeX_Caligraphic-Bold.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Caligraphic-Bold.woff2")
        }
        "KaTeX_Caligraphic-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Caligraphic-Regular.woff2")
        }
        "KaTeX_Fraktur-Bold.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Fraktur-Bold.woff2")
        }
        "KaTeX_Fraktur-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Fraktur-Regular.woff2")
        }
        "KaTeX_Main-Bold.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Main-Bold.woff2")
        }
        "KaTeX_Main-BoldItalic.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Main-BoldItalic.woff2")
        }
        "KaTeX_Main-Italic.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Main-Italic.woff2")
        }
        "KaTeX_Main-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Main-Regular.woff2")
        }
        "KaTeX_Math-BoldItalic.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Math-BoldItalic.woff2")
        }
        "KaTeX_Math-Italic.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Math-Italic.woff2")
        }
        "KaTeX_SansSerif-Bold.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_SansSerif-Bold.woff2")
        }
        "KaTeX_SansSerif-Italic.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_SansSerif-Italic.woff2")
        }
        "KaTeX_SansSerif-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_SansSerif-Regular.woff2")
        }
        "KaTeX_Script-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Script-Regular.woff2")
        }
        "KaTeX_Size1-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Size1-Regular.woff2")
        }
        "KaTeX_Size2-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Size2-Regular.woff2")
        }
        "KaTeX_Size3-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Size3-Regular.woff2")
        }
        "KaTeX_Size4-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Size4-Regular.woff2")
        }
        "KaTeX_Typewriter-Regular.woff2" => {
            include_bytes!("../vendor/katex/fonts/KaTeX_Typewriter-Regular.woff2")
        }
        _ => return not_found_response(),
    };

    asset_response("font/woff2", bytes)
}

pub async fn highlight_css_handler() -> StaticAssetResponse {
    asset_response(
        "text/css; charset=utf-8",
        include_bytes!("../vendor/highlight/highlight.css"),
    )
}

pub async fn highlight_js_handler() -> StaticAssetResponse {
    asset_response(
        "text/javascript; charset=utf-8",
        include_bytes!("../vendor/highlight/highlight.js"),
    )
}

fn asset_response(content_type: &'static str, bytes: &'static [u8]) -> StaticAssetResponse {
    (
        StatusCode::OK,
        [
            (CONTENT_TYPE, content_type),
            (CACHE_CONTROL, CACHE_CONTROL_IMMUTABLE),
        ],
        bytes,
    )
}

fn not_found_response() -> StaticAssetResponse {
    (
        StatusCode::NOT_FOUND,
        [
            (CONTENT_TYPE, "text/plain; charset=utf-8"),
            (CACHE_CONTROL, CACHE_CONTROL_NOT_FOUND),
        ],
        b"Not Found",
    )
}
