# Model calibration

Hashdrills can benchmark the exact provider-neutral [`llm`](https://github.com/simonw/llm)
boundary used by `sample` and `drill`. No collection database is opened.
Omitting the positional case file uses the committed suites in this directory:

```sh
hashdrills eval answers
hashdrills eval generation
```

The older measured results in
[`benchmark-2026-07-31.md`](benchmark-2026-07-31.md) are a historical snapshot,
not a permanent leaderboard or a claim about current providers.

## Choosing a model and run

First confirm the exact IDs and schema support exposed by the installed `llm`
and its plugins:

```sh
llm models --schemas
```

`--model` selects the answer evaluator or generation candidate and defaults to
`gpt-5.6-luna`. It does not select the generation-quality judge. Answer runs
accept `--evaluation-model`, `--evaluation-reasoning-effort`, and repeatable
`--evaluation-llm-option KEY=VALUE`; generation runs use the corresponding
`--generation-*` flags. Shared `--llm-option`, `--schema-mode native|prompt`,
`--llm-executable`, and `--llm-timeout` controls work in both. Hashdrills does
not register missing models: upgrade `llm`, or install and configure the
appropriate provider plugin, if an intended model is absent. At this release,
stable PyPI `llm` 0.31.1 registers neither the Luna candidate default nor the
Sol judge default, so a stable upgrade alone is insufficient: install a
version or provider that registers them, or pass listed schema-capable IDs via
`--model` and, for judged generation, `--judge-model`.

Both commands accept:

- `--repeats N`, from 1 through 10; the default is 1.
- `--concurrency N`, from 1 through 16; the default is 1.
- Repeatable `--case ID` for an exact subset; no `--case` selects the suite.
- `--seed N`; generation mixes it with case ID and repeat number to derive a
  reproducible variation key.
- `--format human|json`; the default is `human`.
- `--llm-timeout SECONDS`, a deadline for every individual call; the default
  is 120 seconds.

For example:

```sh
hashdrills eval answers \
  --evaluation-model openrouter/anthropic/claude-sonnet-4 \
  --repeats 3 \
  --concurrency 4 \
  --format json

hashdrills eval generation \
  --generation-model qwen3:4b \
  --repeats 3 \
  --concurrency 4 \
  --skip-judge
```

The number of answer or candidate calls is selected cases × repeats and is
capped at 1,000 per invocation. A judged generation run can make up to one
additional judge call per usable candidate. Concurrency changes overlap, not
the call count. These commands can therefore consume paid tokens quickly;
start with one or two `--case` selections when calibrating a provider.

## Custom answer cases

Pass a strict bare JSON array to replace the bundled answer suite:

```sh
hashdrills eval answers path/to/answers.json --evaluation-model MODEL_ID
```

Each object has exactly this shape:

```json
[
  {
    "id": "http_patch_wrong",
    "tags": ["http", "semantics", "fail"],
    "question": "Is PATCH guaranteed to be idempotent?",
    "criteria": "No; HTTP does not guarantee PATCH to be idempotent.",
    "response": "Yes, every PATCH can be repeated safely.",
    "allowed_verdicts": ["fail"]
  }
]
```

Allowed verdict strings are `pass`, `partial`, `fail`, `uncertain`, and
`invalid`. `pass` is YEA; every other verdict is NAY. A case may allow multiple
NAY verdicts but may not combine `pass` with a NAY label. Case IDs are unique,
questions and criteria are nonblank, and tags are nonempty. A blank response
is valid for testing invalid or empty learner input.

JSON answer reports include the five-way verdict distribution; binary YEA/NAY
and exact-verdict accuracy; false YEAs and false NAYs; p50, p95, and maximum
latency; feedback character and word counts; first-tag domain summaries; and
case/repeat IDs for misses and errors. Human output presents a concise subset
plus misses and errors.

## Custom generation cases

Pass a strict bare JSON array to replace the bundled generation suite:

```sh
hashdrills eval generation path/to/generation.json --generation-model MODEL_ID
```

Each object has exactly this shape:

```json
[
  {
    "id": "gen_math_product",
    "domain": "math",
    "goal": "Practice exact products from 2 through 12.",
    "question_template": "What is {{choose two integers from 2 through 12 and write their product expression}}?",
    "answer_template": "{{compute the exact product of Q's two integers; digits only}}",
    "judge_requirements": "Q obeys the range, A is exact, and Q does not leak A."
  }
]
```

`goal` may be `null`; every other field is required and nonblank. Both the
question and answer templates must contain at least one valid `{{...}}`
directive. IDs are unique and unknown fields are rejected.

Candidate generation completes before judging begins, so judge traffic does
not distort candidate latency. By default, a blind `gpt-5.6-sol` judge at
reasoning effort `high` checks question validity, target correctness, Q/A
alignment, authored constraints, answer leakage, and atomicity. Override it
with `--judge-model`, `--judge-reasoning-effort`, repeatable
`--judge-llm-option`, and `--judge-concurrency` (1–16). `--skip-judge` reports
generation and diversity without making judge calls.

JSON generation reports include protocol success and errors, candidate
latency, unique-question rate and mean pairwise lexical distance, judge
coverage and latency, judged and end-to-end quality rates, and counts for all
six quality-failure dimensions. Human output presents a concise subset and the
failing case IDs. `--show-samples` also emits each generated question, target,
and rubric for manual inspection. Model judgment is calibration evidence, not
a substitute for reading the generated drills.

## File, privacy, and exit behavior

Custom suites must be regular, non-symlink UTF-8 files no larger than 1 MiB.
Their authored questions, criteria, responses, templates, goals, and judge
requirements are sent to the selected providers. Do not use private cases
with a provider that should not receive them.

Human and JSON reports are content-minimized by default: they omit authored
case text, learner responses, model feedback text, prompt bodies,
provider-option values, and generated Q/A/rubric samples. They still retain
case IDs, first-tag domain names, model IDs, timings, and opaque call-error
text, so inspect them before sharing. `--show-samples` deliberately adds
generated questions, targets, and rubrics to a generation report. Provider and
operating-system logging remain outside the report itself; Hashdrills invokes
`llm` with `--no-log`.

Reports are written to stdout only. They are not inserted into the collection
database or generated-drill archive unless you explicitly redirect or save
their output yourself.

A quality miss is a measured result and does not by itself make the command
fail. Per-call timeouts and partial provider failures are recorded as call
errors; the command fails when answer evaluation, candidate generation, or a
requested judge phase produces zero usable results. Invalid suites or
settings and unknown selected IDs also fail the command.
