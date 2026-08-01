#!/usr/bin/env bash

set -euo pipefail

cargo fmt --all -- --check
cargo check --locked
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo machete
cargo deny check licenses
