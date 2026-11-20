# Offline replay results

Executed on 2026-09-30 with Python 3.12.3.

These are synthetic boundary fixtures, not a representative retrieval or model-quality benchmark.
No embeddings or Jev inference was run. Both providers were skipped in every case.

## Executed checks

- 52 selection and HTTP-emulated adapter tests passed, zero failures/errors/skips.
- Standard-library replay at the default ten-passage ceiling and a tight two-passage ceiling.
- Saved-score replay at a four-passage ceiling, with no new provider calls.
- Black 24.4.0, compilation, dependency consistency, diff checks and production-import isolation passed.

```sh
python -m pytest --noconftest tests/test_context_selection.py --junitxml=.venv/tests.xml
python3 -m evals.context_selection.replay --output .venv/default.json
python3 -m evals.context_selection.replay --max-results 2 --output .venv/tight.json
python3 -m evals.context_selection.replay --max-results 4 --reuse-scores .venv/tight.json --output .venv/reused.json
```

## Measured evidence retention

| Strategy | Result ceiling | Macro evidence recall | Answerable cases missing evidence |
| --- | ---: | ---: | ---: |
| all | Pool size | 1.000 | 0/8 |
| input-topk | 10 | 0.875 | 1/8 |
| bm25-topk | 10 | 0.875 | 1/8 |
| bm25-filter | 10 | 0.583 | 5/8 |
| all | Pool size | 1.000 | 0/8 |
| input-topk | 2 | 0.604 | 5/8 |
| bm25-topk | 2 | 0.417 | 7/8 |
| bm25-filter | 2 | 0.417 | 7/8 |

The same ten cases and 82 passages are used by every strategy. Eight cases have
annotated answer evidence; two have none. All strategies share the character budget.
Required evidence labels are never sent to scorers.

The tight budget cannot preserve every separately located fact in all multi-facet
cases. Lexical rankings also favor some keyword-heavy distractors. Less context
is not sufficient evidence of an improvement, and no winner is selected here.

Corpus SHA-256:
```text
38ae2d1b1c35398052e2cc414197d7d49dda13e1802483abf8ab8724d036c0ed
```

## Deferred

Real embeddings/Jev requests, provider calibration, representative authorized
candidate pools, generated-answer evaluation, provider latency/cost measurements,
and database/extraction integration have not been run.

Production behavior is unchanged. The focused run bypassed application conftest;
it does not stand in for the full application CI suite.
