# Changelog

All notable changes to Hashdrills will be documented in this file.

## Unreleased

- Add a rerunnable `hashdrills setup` wizard with persistent non-secret model
  defaults, readiness checks, connection tests, and deterministic reset and
  configuration flags.
- Default generation and answer evaluation to the benchmarked
  `gpt-5.5-2026-04-23` model with `none` reasoning effort while keeping custom
  providers free of implicit reasoning options.
- Make the conditional-probability example demonstrate inline and display
  KaTeX.

## 0.1.0 — 2026-07-31

- Introduce plaintext generative drills with FSRS scheduling and a focused web review UI.
- Support provider-neutral `llm` model configuration, prompt customization, and separate generation and evaluation settings.
- Add composable deck and path selection with include, union, and exclusion filters.
- Add provider-neutral answer and generation evaluation commands with bundled or custom suites, latency metrics, and blind quality judging.
- Add optional, immutable Markdown archives of generated drills; generated drills are not exported by default.
- Render Markdown, KaTeX, highlighted code, images, audio, video, and source links.
- Add session and collection statistics, undo, secure remote access, and local media serving.
- Add an opt-in Markdown session transcript download from the completion screen.
- Include generation and evaluation benchmarks spanning varied subjects and difficulty levels.
