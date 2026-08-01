# Hashdrills

Plaintext spaced repetition for questions that should change. Inspired by and partially derived from [Hashcards](https://github.com/eudoxia0/hashcards) by [Fernando Borretti](https://borretti.me/article/hashcards-plain-text-spaced-repetition).

Flashcards repeat a fixed prompt, making a stable fact or response available. 
Hashdrills keeps the skill fixed and asks an LLM to generate a fresh question each time. 
You write the goal, the question pattern, and the grading criteria.

> **Status:** Hashdrills is an experimental v0.1 release. Its file format and
> command-line interface may change.

A few principles:
- **Local-first.** Your drills, schedule, and history live on your machine.
- **Markdown.** Write, search, version, back up, and move drills as Markdown files.
- **Model-independent.** Use any provider supported by [`llm`](https://github.com/simonw/LLM), including local models.
- **Inspectability & Control.** Generated questions are disposable by default, sessions can be exported, and AI judgments can be overridden.


## Quick start

Hashdrills requires Rust 1.97 or newer and [Simon Willison's `llm` CLI][llm].

```sh
cargo install --git https://github.com/sangddn/hashdrills --locked
uv tool install llm
llm keys set openai
```

Create a directory for your drills and add `Multiplication.md`:

```md
+++
name = "Bounded multiplication"
+++

G: Practice exact products of two positive integers from 2 through 12.
Q: What is {{a multiplication expression with exactly two integer operands from 2 through 12}}?
A: The response must be mathematically equal to {{the exact integer product of the operands generated in Q}}.
```

Then check, preview, and practice it:

```sh
hashdrills check drills
hashdrills sample drills --count 3
hashdrills drill drills
```

The default model is `gpt-5.5-2026-04-23` with reasoning effort `none`.
Run `llm models --schemas` to see the models available in your installation,
or pass another one with `--model`.

The repository also includes a varied [example collection](example/):

```sh
hashdrills check example
hashdrills sample example --count 1
hashdrills drill example
```

## Drill files

Hashdrills recursively reads Markdown files in a collection directory. A file
may start with TOML metadata:

```md
+++
name = "HTTP semantics"
source = "https://developer.mozilla.org/en-US/docs/Glossary/Idempotent"
+++
```

- `name` sets the deck name shown in the app.
- `source` adds a source link beside the deck name.

Hashdrills does not fetch the source page or send it to the model.

Each drill has `Q:` and `A:`, with an optional `G:` before them:

| Field | Meaning |
| --- | --- |
| `G:` | Optional goal. Hidden while answering; expandable on the result screen. |
| `Q:` | The question shown to the learner. |
| `A:` | The expected answer or grading criteria. |

Fields may span several lines. Tags must begin at column zero. Separate
multiple drills with `---`:

```md
Q: In HTTP semantics, what does idempotent mean?
A: Repeating the same request has the same intended effect as making it once.

---

Q: Is DELETE idempotent under HTTP semantics? Give a brief reason.
A: Yes. Repeating DELETE has the same intended effect on server state.
```

### Generated text

Text inside `{{...}}` in Q or A tells the model what to generate. Everything
outside the braces stays unchanged.

```md
Q: Translate {{one present-tense French sentence of 5–8 words}} into English.
A: Preserve its actors, action, tense, polarity, and concrete details.
```

The model fills all Q and A instructions in one call, so an answer instruction
may refer to values generated for its question. A drill may also be completely
static.

Use `\{{` and `\}}` when you need literal braces. Directives cannot be
empty or nested.

### Markdown, math, and media

Questions, goals, and criteria support sanitized Markdown, including tables,
lists, links, code, and emphasis. Raw HTML is displayed as text.

Use `$...$` for inline KaTeX and `$$...$$` for display math. The
`\(...\)` and `\[...\]` forms work too. Optional commands belong in
`macros.tex` at the collection root.

```md
Q: Evaluate:

$$
\int_0^1 2x\,dx
$$
A: $1$.
```

Local images, audio, and video use ordinary Markdown:

```md
![Free-body diagram](Images/free-body.svg)
![](@/Audio/pronunciation.ogg)
![](@/Video/demonstration.mp4)
```

Relative paths start beside the drill file. `@/` starts at the collection
root. Media must remain inside the collection. Mermaid is not currently
supported; use a local image or SVG for diagrams.

### Writing useful drills

Keep one drill focused on one operation at a reasonably consistent difficulty.
Every generated version should deserve the same grading criteria and review
schedule.

Use `sample` before practicing a new drill. If the previews test substantially
different skills, narrow the instructions or split the drill.

Editing G, Q, or A resets that drill's schedule. Renaming or moving the file
does not.

## Commands

| Command | Purpose |
| --- | --- |
| `hashdrills check [PATH]` | Validate a collection or one Markdown file without calling a model. |
| `hashdrills sample [PATH] --count N` | Preview generated questions and answers without changing review history. |
| `hashdrills drill [COLLECTION]` | Start a review session in the local web app. |
| `hashdrills due [COLLECTION] [DATE]` | List drills due on a date. |
| `hashdrills stats [COLLECTION]` | Show collection and review statistics. |
| `hashdrills eval answers [CASES.json]` | Test answer grading against labelled cases. |
| `hashdrills eval generation [CASES.json]` | Test generated-question quality and latency. |

`due` accepts `today`, `tomorrow`, or `YYYY-MM-DD`. `sample` and
`stats` support `--format human|json`.

Review history lives in `hashdrills.db` inside the collection directory.
Always use the same collection root for `drill`, `due`, and `stats`.

### Selecting drills

Selection flags may be repeated:

- `--include-deck NAME` selects an exact deck name. `--from-deck` is an
  alias.
- `--include-path PATH` selects a file or directory inside the collection.
- `--exclude-deck NAME` and `--exclude-path PATH` remove matches.
- With no include flags, Hashdrills starts with the whole collection.

```sh
hashdrills drill Decks \
  --include-deck Physics \
  --include-deck Literature \
  --include-path Skills \
  --exclude-path Skills/Archive
```

`--new-drill-limit N` limits unseen drills added to a session.
`--drill-limit N` limits the whole session. The Hashcards-compatible aliases
are `--new-card-limit` and `--card-limit`.

## Models and providers

Hashdrills uses the model IDs and provider plugins configured in `llm`.

The default generation and grading model is `gpt-5.5-2026-04-23` with
reasoning effort `none`. Other models use their provider's default reasoning
setting unless you pass a reasoning flag.

```sh
hashdrills drill Decks \
  --model gpt-5.5-2026-04-23 \
  --generation-reasoning-effort none \
  --evaluation-reasoning-effort none
```

Generation and grading may use different models:

```sh
hashdrills drill Decks \
  --generation-model qwen3:4b \
  --evaluation-model gpt-5.5-2026-04-23 \
  --evaluation-reasoning-effort none
```

Reasoning values are `none`, `low`, `medium`, `high`, and `xhigh`.
Actual support depends on the selected model.

### OpenRouter and Ollama

```sh
# OpenRouter
llm install llm-openrouter
llm keys set openrouter
llm openrouter refresh
hashdrills drill Decks --model openrouter/anthropic/claude-sonnet-4

# Ollama
llm install llm-ollama
ollama pull qwen3:4b
hashdrills drill Decks --model qwen3:4b
```

The same pattern works with other `llm` plugins: install the plugin, configure
it, and pass one of the model IDs shown by `llm models`.

Provider options use repeatable `KEY=VALUE` flags:

```sh
hashdrills drill Decks \
  --llm-option temperature=0.2 \
  --generation-llm-option temperature=0.7
```

Use the provider's key store or environment variables for credentials; command
arguments may be visible to other local processes.

If a plugin can produce JSON but does not support `llm` schemas, try
`--schema-mode prompt`.

### Custom prompts

Add short instructions with:

- `--generation-instructions` or `--generation-instructions-file`
- `--evaluation-instructions` or `--evaluation-instructions-file`

Advanced users can replace a stage's prompt with
`--generation-prompt-template-file` or
`--evaluation-prompt-template-file`. A template must contain
`{{input_json}}`. Include `{{default_instructions}}` if you want to retain
Hashdrills' built-in instructions.

### Evaluating a model

Hashdrills includes answer-grading and question-generation test sets:

```sh
hashdrills eval answers \
  --evaluation-model gpt-5.5-2026-04-23 \
  --evaluation-reasoning-effort none

hashdrills eval generation \
  --generation-model gpt-5.5-2026-04-23 \
  --generation-reasoning-effort none \
  --judge-model gpt-5.6-sol \
  --judge-reasoning-effort high
```

The generation judge is configured separately. If `gpt-5.6-sol` is not
available in your `llm` installation, pass another model that supports
structured JSON.

See [`evals/README.md`](evals/README.md) for custom case formats, repeats,
concurrency, output fields, and the included benchmark snapshot.

## Review sessions

1. Write an answer. `Enter` submits; `Shift+Enter` adds a newline.
2. Rate your recall with `1` Forgot, `2` Hard, `3` Good, or `4` Easy.
3. Forgot skips the model check. The other ratings ask the model whether the
   answer meets your criteria.
4. If the model disagrees, accept Forgot or click **Keep my rating**.

`U` undoes the last completed review. The model checks correctness but does
not choose Hard, Good, or Easy.

The completion screen shows session and collection statistics, review history,
a heatmap, and a transcript download. Use **Shutdown** to stop the local
server cleanly.

## Saving session output

Hashdrills always stores review history in `hashdrills.db`. The following
Markdown exports are optional.

### Generated drills

`--save-generated DIR` saves the resolved G, Q, and A from completed generated
reviews when the server shuts down:

```sh
hashdrills drill Decks --save-generated /path/to/private/hashdrills-generated
```

Use a directory outside the active collection. These files contain generated
questions, source links, timestamps, and model metadata, but not the learner's
answer.

### Session transcript

After the session is complete, **Download transcript** creates a Markdown log
containing questions, learner answers, ratings, criteria, model judgments,
overrides, sources, and timing. Hashdrills does not download it automatically.

Both exports contain private learning data. Inspect them before sharing.

## Reviewing from another device

The web app listens on `127.0.0.1:8000` by default. Each launch prints a new
authenticated URL.

To open it from a phone over Tailscale, bind to the computer's Tailscale
address:

```sh
hashdrills drill Decks \
  --host 100.64.0.7 \
  --port 8000 \
  --open-browser false
```

Open the exact printed URL on the phone. Treat it like a password: anyone with
it can control the session. `--no-auth` is available only for loopback access.
Use HTTPS when the network itself is not trusted.

## Privacy

The configured model receives the goal, question, criteria, generated text,
and learner answer needed for its call. Do not put secrets in a drill unless
you are comfortable sending them to that provider.

Hashdrills runs `llm` with `--no-log`, but keeps its own private
`hashdrills.db` for scheduling and review history. A local-model plugin can
keep model traffic on your machine.

Images, audio, and video are shown in the app but are not sent to the current
text-only generation or grading model.

## License

Hashdrills is derived from [Fernando Borretti's Hashcards][hashcards] and is
licensed under Apache License 2.0. See [`NOTICE`](NOTICE), [`LICENSE`](LICENSE),
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

[hashcards]: https://github.com/eudoxia0/hashcards
[llm]: https://github.com/simonw/llm
