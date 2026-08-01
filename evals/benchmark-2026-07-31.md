# Model benchmark — 2026-07-31

This snapshot compares semantic answer evaluation and drill generation. It is
calibration evidence for defaults, not a permanent leaderboard.

## Method

- Evaluation used 60 synthetic, human-labeled cases spanning math, physics,
  chemistry, biology, programming, history, literature, visual art, language,
  and practical skills. The primary metric is the product decision: YEA
  (`pass`) versus NAY (all other verdicts).
- Generation used 12 synthetic drill specifications from similarly varied
  domains. Finalists generated each case three times at concurrency 1: 36
  generations per configuration.
- Direct-API latency is wall time through the complete response body, JSON
  decoding, production parsing, and—during generation—template rendering.
- Generation quality was judged after all candidate calls, so judge traffic did
  not affect candidate latency. `gpt-5.6-sol@high` performed the blind structured
  review, followed by manual inspection of outputs and misses.
- The Sol judge is useful but not an oracle: manual inspection caught at least
  one false pass. Final quality conclusions therefore use both signals.

## Evaluator

The corrected three-repeat `EVAL_SUMMARY` result was:

| Model | Effort | Decision-correct | Exact five-way | False pass | False NAY | Errors | Mean | p50 | p95 | Max | Feedback chars | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.5-2026-04-23` | `none` | 180/180 | 160/180 (88.89%) | 0 | 0 | 0 | 1.247s | 0.963s | 1.766s | 10.404s | 20.961 | 18.206 |

The initial screen also found `gpt-5.5@low` decision-correct on 180/180,
whereas `gpt-5.4-mini@low` produced one conservative false NAY in 180 calls.
Luna's effort sweep was order/provider-load confounded and did not justify a
claim that greater reasoning effort reduced latency.

Earlier reported `gpt-5.5@none` latency values (0.913s p50 and 1.414s p95)
stopped the clock before the complete body and production parse. They are
superseded by the corrected figures above; their quality counts remain
consistent with the rerun.

## Generator

The final benchmark added a per-attempt variation key to the otherwise identical
production prompt. `GEN_SUMMARY` results, adjudicated with Sol plus manual
inspection, were:

| Model | Effort | Quality | Generation/judge errors | p50 | p95 | Max | Unique questions | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.5-2026-04-23` | `none` | 36/36 | 0 | 1.231s | 2.354s | 10.389s | 28/36 (77.8%) | 42.72 |
| `gpt-5.6-terra` | `high` | 36/36 | 0 | 1.712s | 3.132s | 5.215s | 27/36 (75.0%) | 120.03 |

For context, the immediately preceding run had no variation key:

| Model | Effort | Quality | p50 | p95 | Unique questions |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.5-2026-04-23` | `none` | 36/36 | 1.312s | 1.874s | 22/36 (61.1%) |
| `gpt-5.6-terra` | `high` | 35/36 | 1.635s | 2.836s | 23/36 (63.9%) |

The variation key materially improved diversity without reducing adjudicated
quality. `gpt-5.5@none` matched Terra's final quality with lower median/tail
latency and roughly one-third of its output tokens.

## Production smoke

After a fresh debug rebuild, two sequential real generations through the `llm`
subprocess completed in 4.21s total. This verifies the production path; it is a
two-sample smoke test and must not be compared directly with the direct-API
latency tables.

## Interpretation at the time

This snapshot supported `gpt-5.5-2026-04-23@none` as its comparison baseline
for both evaluation and generation. It does not describe the current product
default, which may change as models and provider plugins evolve. Generation
and evaluation controls remain independent, and the benchmark keeps the
workloads separate so model choices can diverge when evidence warrants it.

These fixtures are synthetic and small. Live results are observations rather
than golden assertions: inspect `EVAL_MISS`, `GEN_MISS`, and `GEN_SAMPLE`
records, revisit human labels, and rerun after meaningful prompt, model, or API
changes.
