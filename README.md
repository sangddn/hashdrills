# Hashdrills

Plaintext generative spaced practice.

A Hashcard helps make a stable fact or response available. Hashdrills schedules
a stable **practice target**, then generates a fresh, comparable question each
time it comes due. The generated question is an instance of the target; it is
not the thing being scheduled.

> **Status:** Hashdrills is an experimental v0.1 release. Its file and CLI
> surfaces may still change. Generated drills and model judgments always need
> human inspection and calibration.

## A first drill

Each Markdown file contains one or more drills:

```md
+++
name = "Bounded multiplication"
source = "https://en.wikipedia.org/wiki/Multiplication"
+++

G: Practice exact products of two positive integers from 2 through 12.
Q: What is {{a multiplication expression with exactly two integer operands from 2 through 12}}?
A: The response must be mathematically equal to {{the exact integer product
of the operands generated in Q}}. Equivalent exact notation is acceptable.
```

`G:` is the optional hidden goal. `Q:` and `A:` are templates. Text inside
`{{...}}` is an instruction for generated text; everything else is literal.
Hashdrills resolves the Q and A directives together, then freezes the concrete
question and target before the learner answers.

See [`example/`](example/) for a bounded arithmetic drill and a neutral,
source-bounded classification drill.

## Install

Hashdrills is a Rust command-line application requiring Rust 1.97 or newer.
Install it directly from the public GitHub repository:

```sh
cargo install --git https://github.com/sangddn/hashdrills --locked
```

Or clone it first for development:

```sh
git clone https://github.com/sangddn/hashdrills.git
cd hashdrills
cargo install --path . --locked
```

Hashdrills uses [Simon Willison's `llm` CLI][llm] as its model boundary. Install
`llm` separately, then let `llm` manage provider plugins, keys, model IDs, and
aliases:

```sh
uv tool install llm
llm keys set openai
llm models
```

From a source checkout, try the included example collection:

```sh
hashdrills check example
hashdrills sample example --count 3
hashdrills drill example
```

The default model is `gpt-5.6-luna`. Confirm that the installed `llm` and its
plugins know the models and schemas you intend to use:

```sh
llm models --schemas
```

Hashdrills uses the registry exactly as `llm` exposes it; it does not add
models or modify your `llm` configuration. At this release, stable PyPI `llm`
0.31.1 does not register `gpt-5.6-luna` or `gpt-5.6-sol`, so upgrading within
that stable release alone does not make the defaults available. Until a
released `llm` version or provider plugin registers them, install a version or
plugin that does, or explicitly choose a listed schema-capable model with
`--model MODEL_ID` (and `--judge-model MODEL_ID` for a generation eval).

Generation and answer evaluation do not set a reasoning option unless you
request one. This keeps plugins that do not implement `reasoning_effort`
usable. The blind generation-eval judge is the exception: it defaults to
`high`. Supported CLI values are `none`, `low`, `medium`, `high`, and `xhigh`,
but the chosen model and plugin decide which values actually work:

```sh
hashdrills drill example \
  --generation-reasoning-effort low \
  --evaluation-reasoning-effort none
```

### Providers and local models

There is intentionally no Hashdrills `--provider` switch. Install and
configure a provider in `llm`, confirm its exact model ID with `llm models`,
then pass that ID or an `llm` alias to `--model`.

OpenAI support ships with `llm`:

```sh
llm keys set openai
hashdrills sample example --model gpt-5.6-luna --count 3
```

For OpenRouter:

```sh
llm install llm-openrouter
llm keys set openrouter
llm openrouter refresh
hashdrills drill example --model openrouter/anthropic/claude-sonnet-4
```

For a local Ollama model:

```sh
llm install llm-ollama
ollama pull qwen3:4b
hashdrills sample example --model qwen3:4b --count 3
```

The same pattern works with `llm-gguf`, `llm-mlx`, or a locally hosted
OpenAI-compatible endpoint: register the model with `llm`, then use its listed
ID. Native `llm` schema support is the default. If a plugin can return JSON but
does not implement native schemas, ask Hashdrills to put the exact schema in
the prompt instead:

```sh
hashdrills drill example \
  --model YOUR_LOCAL_MODEL_ID \
  --schema-mode prompt
```

Both modes still use the same strict local JSON parser; `prompt` mode cannot
make an incapable model reliably follow a schema.

Provider-specific model options are repeatable `KEY=VALUE` arguments. Shared
options apply to both calls; stage-specific options with the same key win:

```sh
hashdrills drill example \
  --llm-option temperature=0.2 \
  --generation-llm-option temperature=0.7 \
  --evaluation-llm-option temperature=0
```

Option names and values belong to the selected `llm` plugin. They are passed
as process arguments, so do not put API keys, bearer tokens, or other secrets
in them. Use `llm keys set`, environment variables, or the plugin's own
configuration instead.

## Format

Hashdrills recursively reads Markdown files in a collection directory.

### File metadata

A file may begin with TOML frontmatter delimited by `+++`:

```toml
+++
name = "HTTP semantics"
source = "https://developer.mozilla.org/en-US/docs/Glossary/Idempotent"
+++
```

- `name` overrides the file stem as the logical deck name.
- `source` is optional provenance for the whole file.

A source URL records where the drill came from; it is not, by itself, source
content. Put any rule needed to generate or grade the drill in its compact
executable contract, or use a workflow that explicitly loads the source.

When present, it appears in the review UI as **Source ↗**. Hashdrills accepts
`http://`, `https://`, and `obsidian://` source URLs; `check` rejects other
schemes.

### Rendered content

Questions, criteria, and goals render as sanitized Markdown, including GFM
tables, lists, emphasis, links, and inline or fenced code. Raw HTML is inert.
Use `$...$` or `\(...\)` for inline KaTeX and `$$...$$` or `\[...\]` for
display math. Optional custom commands can be defined in `macros.tex` at the
collection root.

Local media follows Hashcards' path conventions. A deck-relative path starts
beside the Markdown file; `@/` starts at the collection root:

```md
![Free-body diagram](Images/free-body.svg)
![](@/Audio/pronunciation.ogg)
![](@/Video/demonstration.mp4)
```

Supported images are PNG, JPG/JPEG, GIF, SVG, WebP, BMP, and AVIF; supported
audio is MP3, WAV, OGG, M4A, and FLAC; supported video is MP4, WebM, and MOV.
Media must remain inside the collection.
External image URLs and escaping paths are rejected. Hashdrills serves accepted
media through the same origin as the review UI, so it also works when the UI is
opened from another device over Tailscale; no `file://` access is required.

Mermaid is not yet supported. Use a local PNG or SVG for diagrams for now.
Visual media is display-only: the current text model does not see image or
video contents during generation or evaluation.

### Drill blocks

A drill is `[G:] Q: A:` in that order. Tags begin at column zero, fields may
span lines, and `---` separates drills:

```md
G: Optional hidden goal and invariant.
Q: Question template.
A: Answer target or grading criteria.

---

Q: A second, fully static question.
A: Its static target.
```

- `G:` is optional, plain text, and hidden from the learner while answering.
  It does not expand `{{...}}` directives.
- `Q:` is the displayed question template.
- `A:` is the hidden concrete answer or acceptance criteria used for feedback.
  Its frozen rendered text is authoritative; Hashdrills does not invent a
  second grading intention behind the author's criteria.

The goal is part of the executable drill definition, not a place for project
history or a long pedagogical essay. Keep broader intentions in the source
note that owns the practice.

### Generation directives

`{{instruction}}` marks generated text in Q or A. Q and A directives are
resolved jointly, so an answer can refer to facts generated into its question.
Literal text outside directives is preserved.

```md
Q: Translate {{one present-tense French sentence of 5–8 words}} into English.
A: Preserve its actors, action, tense, polarity, and concrete details. Natural
English phrasing may vary.
```

This mixed example generates only part of Q and uses a static rubric in A.
Zero directives is also valid:

```md
Q: In HTTP semantics, what does idempotent mean?
A: Repeating the same request has the same intended effect on the server as
making it once.
```

To write literal delimiters, escape them:

```md
Q: Render these as text: \{{ and \}}.
A: The literal opening and closing directive delimiters.
```

`\{{` renders `{{`, and `\}}` renders `}}`. Directives do not nest; empty,
nested, or unclosed directives are validation errors.

## Atomicity

FSRS schedules the stable G/Q/A template, never a generated instance. A useful
authoring test is:

> Would every valid sample exercise the same operation, at comparable
> difficulty, under the same rubric, and deserve the same future schedule?

If not, split the drill. `{{a multiplication problem}}` is too broad;
`{{two positive integer operands from 2 through 12}}` is bounded enough for an
initial target. Use `sample` to inspect variation and drift before enrolling a
drill.

Meaningful changes to G/Q/A change the target's content-addressed identity in
the v0.1 design. Generated wording, model choice, seed, file path, deck name,
and source provenance do not become scheduler identities.

## Commands

The command surface follows the Hashcards shape:

| Command | Role |
| --- | --- |
| `hashdrills check [PATH]` | Parse and validate a collection or one Markdown file without a model call or database. |
| `hashdrills sample [PATH] --count N` | Generate N previews per matching target, including their hidden targets, without opening the collection database. |
| `hashdrills drill [COLLECTION]` | Practice due and admitted new targets in the local web UI and save confirmed reviews. |
| `hashdrills due [COLLECTION] [DATE]` | Show targets scheduled for a date. |
| `hashdrills stats [COLLECTION]` | Report collection and review statistics. |
| `hashdrills eval answers [CASES.json]` | Benchmark answer judgments against bundled or custom labelled cases. |
| `hashdrills eval generation [CASES.json]` | Benchmark generated drills, with an optional blind quality judge. |

`sample` may contact the configured model provider, but it does not enroll a
target, create attempt history, or change scheduling state. Its default count
is three previews per selected specification, and `--format human|json` makes
it useful for either inspection or tooling. `stats` also supports
`--format human|json`; `due` accepts `today`, `tomorrow`, or `YYYY-MM-DD`.

`check` and `sample` may take one Markdown file. Scheduler-backed commands
(`drill`, `due`, and `stats`) require the canonical collection directory as
their positional path. Keep using that same root so `hashdrills.db` is not
fragmented across nested folders; select one file or subtree with
`--include-path` instead.

### Selecting a practice set

`sample`, `drill`, `due`, and `stats` share composable selectors:

- Repeat `--include-deck NAME` for an exact, case-sensitive union of logical
  deck names. `--from-deck` is an alias.
- Repeat `--include-path PATH` to add one source file or every source file
  below a directory.
- Repeat `--exclude-deck NAME` or `--exclude-path PATH` to subtract matches.
  Exclusions always win.
- With no includes, the candidate set starts as the whole collection. Every
  selector must resolve and match; a typo is an error rather than an empty or
  unexpectedly broad session.

Relative path selectors are resolved from the collection root. They must stay
inside that root; a directory selects all authored decks below it. Name and
path includes form one union, so this reviews Physics, Literature, and every
deck below `Skills`, except the explicit exclusions:

```sh
hashdrills drill Decks \
  --include-deck Physics \
  --include-deck Literature \
  --include-path Skills \
  --exclude-deck Drafts \
  --exclude-path Skills/Archive \
  --new-drill-limit 5 \
  --drill-limit 20
```

`--new-drill-limit` controls how many unseen specifications enter this
session; it is not an authoring quota. `--drill-limit` caps the entire queue.
The Hashcards-compatible aliases are `--new-card-limit` and `--card-limit`.

### Models and prompts

`sample` uses the generation controls below; `drill` uses both generation and
evaluation controls:

- `--model ID` is the fallback for both stages.
- `--generation-model ID` and `--evaluation-model ID` override it for one
  stage, even when the IDs come from different installed `llm` plugins.
- `--generation-reasoning-effort` and `--evaluation-reasoning-effort` are
  optional. Omitting them sends no reasoning option.
- `--llm-timeout SECONDS` sets the per-call timeout; the default is 120.
- `--schema-mode native|prompt` controls how the required response schema is
  conveyed.

For example, a local model can generate while a hosted model judges:

```sh
hashdrills drill Decks \
  --generation-model qwen3:4b \
  --evaluation-model gpt-5.6-luna \
  --generation-llm-option temperature=0.7 \
  --evaluation-reasoning-effort low
```

Add compact trusted guidance without replacing Hashdrills' protocol:

```sh
hashdrills drill Decks \
  --generation-instructions-file path/to/generate-extra.txt \
  --evaluation-instructions 'Accept equivalent exact notation.'
```

Each stage accepts either inline instructions or an instruction file, not
both. For full control, use `--generation-prompt-template-file` or
`--evaluation-prompt-template-file`. An advanced template must contain the
literal `{{input_json}}` placeholder. It may contain
`{{default_instructions}}`; omitting that placeholder intentionally omits the
built-in task instructions. If same-stage inline or file instructions are also
supplied, the template must retain `{{default_instructions}}` so they have a
defined insertion point.

All customization files must be regular UTF-8 files no larger than 64 KiB.
Treat them as trusted code-like configuration: they can weaken generation or
grading behavior, and their contents are sent to the selected provider.

`--llm-executable PATH` selects a different executable, which Hashdrills runs
as local code. Use it only with a program you trust; it is not a safe way to
name a remote provider.

### Evaluating model settings

Use the bundled calibration suites by omitting the positional JSON path:

```sh
hashdrills eval answers \
  --evaluation-model gpt-5.6-luna \
  --evaluation-reasoning-effort low \
  --repeats 3 \
  --concurrency 4

hashdrills eval generation \
  --generation-model qwen3:4b \
  --repeats 3 \
  --concurrency 4 \
  --judge-model gpt-5.6-sol \
  --judge-reasoning-effort high \
  --judge-concurrency 4
```

The commands use the same provider-neutral `llm` settings as `sample` and
`drill`: `--model` selects the answer evaluator or generation candidate, and
`--llm-option` supplies shared provider options. The `--evaluation-*`,
`--generation-*`, and `--judge-*` flags scope overrides to one stage;
`--judge-model` independently defaults to `gpt-5.6-sol`. `--llm-timeout` is a
deadline for each individual model call and defaults to 120 seconds.
`--skip-judge` makes a generation run candidate-only; `--show-samples`
deliberately includes generated Q/A/rubric text in its report.

Both commands accept `--repeats` (1–10), `--concurrency` (1–16), repeatable
exact `--case ID` selection, `--seed`, and `--format human|json`. The seed is
used to derive reproducible generation variation keys. A run is capped at
1,000 answer or candidate calls; judged generation can make up to one
additional judge call per usable candidate.

Reports cover YEA/NAY and exact five-verdict accuracy, false YEAs and NAYs,
latency, feedback length, domain breakdowns, candidate/judge success and
latency, blind-judge quality, and repeated-question diversity. The JSON format
contains the complete structured metrics; human output is a concise subset.
Reports omit authored case text, prompt bodies, provider-option values, and
generated samples by default. Custom suites are strict bare JSON arrays and
are sent to the selected model provider; see
[`evals/README.md`](evals/README.md) for their exact schemas, cost model, and
privacy boundary.

### Saving generated drills

By default, a session writes no standalone Markdown copies of generated
instances. Hashdrills' private `hashdrills.db` still retains frozen instances,
answers, evaluations, accepted grades, and undo history for scheduling and
auditability.

Pass `--save-generated DIR` to additionally archive effective generated
reviews when the server shuts down, whether through **Shutdown** or an
interactive `Ctrl+C`:

```sh
hashdrills drill Decks --save-generated /path/outside/your-collection/hashdrills-generated
```

Each effective, non-static generated review becomes one immutable Markdown
file. Its resolved `G:`, `Q:`, and `A:` body values are copied verbatim for
readability; canonical byte-exact values live in the `resolved_goal`,
`resolved_question`, `resolved_target`, and `resolved_rubric` frontmatter
fields. Frontmatter also records generation and review dates, deck/source
provenance, model/protocol data, the stable spec hash, and local trace IDs. A
review removed with Undo is not in the effective set; an unfinished review is
not archived. The plaintext archive omits the learner response and AI
verdict/comment, which remain in the SQLite audit data.

The archive directory must be outside the active collection root; keep it
outside the collection's repository as well so private generated evidence is
not committed by accident. Collection parsing ignores copied-back archives
only when both exact markers are present:

```toml
hashdrills_archive_kind = "generated_drill"
hashdrills_archive_format_version = 1
```

The output is private evidence, not safe publication material. It contains
generated Markdown, goals, source links, timestamps, and model metadata, while
filenames expose the review date, deck slug, and local IDs. Because the body is
verbatim, generated raw HTML and remote media URLs may become active when an
archive is opened in another renderer; treat it as untrusted content and use a
sanitizing, no-remote-load renderer. On Unix, a newly created final archive
directory gets mode `0700`; Hashdrills does not change an existing directory's
permissions.

Hashdrills preflights the destination at startup but publishes the immutable
files only when shutdown begins. Once publication starts, Undo is sealed so a
partially written batch cannot diverge from the database. If **Shutdown**
reports a partial failure, retry it: already published identical files are
recognized and the operation is idempotent.

### Downloading a session transcript

After **SESSION COMPLETE**, **Download transcript** returns one Markdown
attachment for the current launch before Shutdown. The download is a
user-triggered snapshot governed by the launch access mode (authenticated by
default, or loopback-only when `--no-auth` is explicitly enabled). Hashdrills
does not automatically write it to the filesystem. It is capped at 16 MiB and
served with `private, no-store` response headers.

Entries follow their effective resolution order. An undone review is removed;
a later regrade appears only after the session completes again. Downloading
does not seal Undo, so an earlier download can become stale—download again
after re-completion. The transcript includes frozen questions, learner
answers, self-ratings, effective ratings and override resolution, targets and
criteria, AI verdicts/comments, goals/sources, durations, and model IDs when
available, together with the corresponding model-protocol versions. Forgot
remains a reviewed entry with an intentionally skipped AI check. Ordinary
skips are labelled; a pre-answer skip has no invented answer, an
unresolved-check skip preserves the evidence that existed, and a generation
failure has no invented question.

Dynamic values are placed in variable-length Markdown code fences so they
cannot break the transcript's structure. Newlines and tabs are preserved;
other C0/C1 control characters are visibly escaped. If a transcript is copied
into a collection, discovery ignores it only when this exact supported marker
pair is present:

```toml
hashdrills_session_log_kind = "session_transcript"
hashdrills_session_log_format_version = 1
```

The file is sensitive learning data. Generated Markdown, raw HTML, links, and
media references remain untrusted if extracted or rendered elsewhere; inspect
it before sharing and use a sanitizing renderer with remote loads disabled.

### Browser access

`drill` binds to `127.0.0.1:8000` and opens a browser by default. Every launch
uses a fresh authenticated URL. Opening it exchanges the bearer token for a
host-only, HttpOnly session cookie; the capability expires when the process
exits, but the query-token URL remains valid for that launch. Anyone who has it
can control the session: do not paste it into chats or link previewers, and do
not retain it in reverse-proxy access logs.

To review from a phone on Tailscale, prefer binding the machine's concrete
Tailscale address, disable automatic local opening if desired, and open the
exact URL printed by Hashdrills on the phone:

```sh
hashdrills drill Decks \
  --host 100.64.0.7 \
  --port 8000 \
  --open-browser false
```

A wildcard bind also listens on other available network interfaces. If that is
intentional, it requires an explicit public origin so Hashdrills can validate
browser requests and print the correct remote URL:

```sh
hashdrills drill Decks \
  --host 0.0.0.0 \
  --port 8000 \
  --public-url http://100.64.0.7:8000 \
  --open-browser false
```

`--public-url` must be an `http` or `https` origin without credentials, a path,
query, or fragment. `--no-auth` is accepted only with a loopback bind; it
cannot disable authentication on a LAN, Tailscale, or wildcard listener, and
is intentionally weaker even on loopback. Plain HTTP is appropriate only over
a trusted encrypted tunnel such as Tailscale; use HTTPS on an untrusted
network. A reverse proxy must preserve the configured Host and public Origin.

## Evaluation and scheduling

Hashdrills freezes Q and A together, then runs this review loop:

1. Write an answer. `Enter` submits; `Shift+Enter` inserts a newline.
2. Self-rate with `1`–`4`, or click/tap: Forgot, Hard, Good, or Easy.
3. Forgot is persisted immediately and skips evaluation. Hard, Good, and Easy
   are persisted before the AI checks whether the answer is acceptable.
4. Yea accepts the learner's grade. Nay offers **Continue as Forgot**;
   **Keep my rating** is a deliberate click/tap override with no shortcut.
   Pressing `Enter` continues with the displayed primary action.

The AI checks acceptability; it never selects Hard, Good, or Easy. Uncertain
or invalid results do not update FSRS. Provider latency occurs only when the
learner claims success. `U` undoes the most recently accepted review while
preserving its audit history.

The completion screen separates reviewed drills from skips, reports retention
from accepted grades and average wall-clock duration per resolved drill, shows
active collection totals, and keeps a 53-week review heatmap plus grade and
scheduling graphs inside **History**. It also offers the opt-in transcript
download described above. **Shutdown** saves the finished state and gracefully
stops the local server.

Generated questions, answers, self-ratings, evaluations, and undo events are
retained with enough provenance to audit what happened. The stable template
remains the scheduled object.

## Privacy

Generation and evaluation send G, Q, A, the generated instance, and the
learner's response to the model configured through `llm`. File-level `source`
is provenance, not model context in v0.1. Do not put secrets or material you
would not send to that provider in a drill. A local-model plugin can reduce
provider exposure, subject to model support.

`llm` logs ordinary prompts and responses to its own SQLite database by
default. Hashdrills invokes it with `--no-log` for each model call. As optional
defense in depth, you can also disable that separate log globally:

```sh
llm logs off
```

Hashdrills' own collection database retains scheduling and attempt evidence.
Treat it as private learning data, especially when answers
are free-form. `--save-generated` creates an additional private corpus with
the metadata described above; it is never enabled implicitly. A downloaded
session transcript is another private copy and may contain learner answers and
AI comments that the generated-drill archive deliberately omits.

## Lineage and license

Hashdrills is derived from [Fernando Borretti's Hashcards][hashcards] and
follows its plaintext collections, FSRS scheduling, local web UI, and command
shape. Because this repository contains modified Apache-2.0 Hashcards code, it
is distributed under Apache License 2.0 rather than MIT-only. See
[`NOTICE`](NOTICE), [`LICENSE`](LICENSE), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

[hashcards]: https://github.com/eudoxia0/hashcards
[llm]: https://github.com/simonw/llm
